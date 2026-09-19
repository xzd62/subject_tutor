"""长期记忆工具：memory_read / memory_write（由模型自主调用）。"""

from __future__ import annotations

from typing import Literal

from langchain.tools import tool

from ..memory import MemoryError, MemoryStore

MemoryType = Literal["学生画像", "错题与错因", "讲解偏好", "薄弱知识点"]
MemoryWriteMode = Literal["append", "replace"]


def make_memory_tools(store: MemoryStore) -> list:
    @tool
    def memory_read(name: str) -> str:
        """查看一条长期记忆的完整内容。当系统提示的记忆索引里有相关条目时使用。

        name: 记忆名称，必须是索引中出现的名称。
        """
        try:
            text = store.read_text(name)
        except MemoryError as exc:
            return f"读取失败：{exc}"
        if text is None:
            names = [meta.name for meta in store.list_metas()]
            available = "、".join(names) if names else "（暂无记忆）"
            return f"未找到记忆「{name}」。可用记忆：{available}"
        return text

    @tool
    def memory_write(
        name: str,
        type: MemoryType,
        description: str,
        content: str,
        mode: MemoryWriteMode = "append",
    ) -> str:
        """新增或更新一条长期记忆，用于跨对话记住学生的情况。

        name: 记忆名称，与文件名一致（如「学生画像」「数学错题本」）。
        type: 记忆类别，四选一。
        description: 一句话描述这条记忆，会出现在系统提示的记忆索引里。
        content: 记忆正文（markdown）。
        mode: append=在已有记忆后追加（默认，适合错题、薄弱点）；replace=用 content 整体覆盖正文（适合更新画像、偏好）。
        """
        try:
            meta, created = store.write(name, type, description, content, mode)
        except MemoryError as exc:
            return f"写入失败：{exc}"
        action = "创建" if created else "更新"
        return f"已{action}记忆「{meta.name}」（{meta.type}），更新时间 {meta.updated_at}"

    return [memory_read, memory_write]
