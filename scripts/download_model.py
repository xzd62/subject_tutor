"""预下载嵌入模型到项目 models 目录。

默认走 hf-mirror 并关闭 HuggingFace Xet（本机网络下 Xet CAS 会挂起）。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_NAME = os.environ.get("TUTOR_RAG_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


def main() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_NAME, max_workers=4)
    print(f"模型已就绪: {path}")


if __name__ == "__main__":
    main()
