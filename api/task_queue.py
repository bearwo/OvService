from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import MAX_CONCURRENT


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None


class TaskQueue:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=16)
        self._tasks: dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(self) -> str:
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id)
        self._tasks[task_id] = task
        await self._queue.put(task_id)
        return task_id

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def set_running(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.RUNNING

    def set_completed(self, task_id: str, result: Any) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.COMPLETED
            self._tasks[task_id].result = result

    def set_failed(self, task_id: str, error: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.FAILED
            self._tasks[task_id].error = error

    def cancel(self, task_id: str) -> bool:
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                return True
        return False

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()
