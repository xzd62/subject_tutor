"""可观测性：统一日志、分阶段计时、带百分比的运行报告。

开发期的原则是所有阶段都留痕：
- 每个阶段记录耗时、计数、比例指标、异常备注；
- 转换回退、超长块、无 overlap 等异常都作为 event 落盘；
- 每次运行输出 outputs/reports/run-<id>.json，便于复盘与回归对比。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

LOGGER_NAME = "tutor_rag"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logger: logging.Logger | None = None


def force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    force_utf8_console()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    if _logger is not None:
        return _logger
    return setup_logging()


def pct(numerator: float, denominator: float, digits: int = 1) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.{digits}f}%"


def ratio_value(numerator: float, denominator: float) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "pct": round(100.0 * numerator / denominator, 2) if denominator else None,
    }


@dataclass
class StageMetrics:
    name: str
    duration_s: float = 0.0
    values: dict[str, Any] = field(default_factory=dict)
    ratios: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def ratio(self, key: str, numerator: float, denominator: float) -> None:
        self.ratios[key] = ratio_value(numerator, denominator)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def lines(self) -> list[str]:
        parts = [f"[{self.name}] 耗时 {self.duration_s:.2f}s"]
        for key, value in self.values.items():
            parts.append(f"    {key}: {value}")
        for key, value in self.ratios.items():
            num, den, p = value["numerator"], value["denominator"], value["pct"]
            pct_text = "n/a" if p is None else f"{p:.1f}%"
            parts.append(f"    {key}: {num}/{den} ({pct_text})")
        for note in self.notes:
            parts.append(f"    ! {note}")
        return parts


@dataclass
class RunReport:
    command: str
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    )
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str = ""
    status: str = "running"
    stages: list[StageMetrics] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageMetrics]:
        metrics = StageMetrics(name=name)
        self.stages.append(metrics)
        logger = get_logger()
        logger.info("▶ 阶段开始: %s", name)
        start = time.perf_counter()
        try:
            yield metrics
        except Exception as exc:
            metrics.note(f"ERROR: {exc!r}")
            self.event("stage_error", stage=name, error=repr(exc))
            self.status = "failed"
            raise
        finally:
            metrics.duration_s = time.perf_counter() - start
            logger.info("■ 阶段结束: %s (%.2fs)", name, metrics.duration_s)

    def event(self, kind: str, level: str = "warning", **data: Any) -> None:
        record = {
            "kind": kind,
            "level": level,
            "time": datetime.now().isoformat(timespec="seconds"),
            **data,
        }
        self.events.append(record)
        logger = get_logger()
        detail = " ".join(f"{k}={v}" for k, v in data.items())
        getattr(logger, level if level in {"debug", "info", "warning", "error"} else "warning")(
            "event[%s] %s", kind, detail
        )

    def finish(self, status: str = "success") -> None:
        self.status = status
        self.finished_at = datetime.now().isoformat(timespec="seconds")

    def stage_by_name(self, name: str) -> StageMetrics | None:
        for stage in reversed(self.stages):
            if stage.name == name:
                return stage
        return None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def summary(self) -> str:
        header = f"运行报告 {self.run_id} [{self.command}] 状态={self.status}"
        rows = [header]
        for stage in self.stages:
            rows.extend(stage.lines())
        warnings = [e for e in self.events if e.get("level") != "info"]
        if warnings:
            rows.append(f"事件: {len(warnings)} 条（见报告 JSON 的 events）")
            for event in warnings[:10]:
                detail = " ".join(
                    f"{k}={v}" for k, v in event.items() if k not in {"kind", "level", "time"}
                )
                rows.append(f"    - {event['kind']}: {detail}")
        return "\n".join(rows)

    def log_summary(self) -> None:
        logger = get_logger()
        for line in self.summary().splitlines():
            logger.info(line)


class Progress:
    """按百分比打点的进度条，默认每 10% 输出一次。"""

    def __init__(self, total: int, label: str, every: float = 0.1):
        self.total = max(total, 0)
        self.label = label
        self.every = every
        self._next = every

    def tick(self, index: int, **info: Any) -> None:
        if self.total <= 0:
            return
        fraction = index / self.total
        if fraction + 1e-9 >= self._next or index >= self.total:
            detail = " ".join(f"{k}={v}" for k, v in info.items())
            get_logger().info(
                "%s %d/%d (%.1f%%) %s",
                self.label, index, self.total, 100.0 * index / self.total, detail,
            )
            while self._next <= fraction + 1e-9:
                self._next += self.every
