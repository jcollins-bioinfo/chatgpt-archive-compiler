"""Small process-local job runner for a single-user local Dash deployment."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    """Lifecycle states exposed without private exception content."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LogStatus(StrEnum):
    """Accessible state for one append-only progress entry."""

    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETE = "complete"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True)
class JobLogEntry:
    """Content-free operation record retained after completion."""

    elapsed_seconds: float
    stage: str
    description: str
    status: LogStatus
    duration_seconds: float | None = None
    count: int | None = None


@dataclass
class JobSnapshot:
    """Content-free mutable state guarded by the registry lock."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = "Waiting for local worker"
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    log: list[JobLogEntry] = field(default_factory=list)


class JobRegistry:
    """Run bounded local work away from the Dash request callback."""

    def __init__(self, maximum_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=maximum_workers, thread_name_prefix="atlas")
        self._jobs: dict[str, JobSnapshot] = {}
        self._lock = threading.Lock()

    def submit(self, operation: Callable[[Callable[[str], None]], Any]) -> str:
        """Queue an operation receiving a stage callback and return immediately."""

        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobSnapshot(
                job_id=job_id,
                log=[
                    JobLogEntry(
                        0, "Local worker", "Waiting for the private local worker.", LogStatus.QUEUED
                    )
                ],
            )

        def execute() -> None:
            with self._lock:
                job = self._jobs[job_id]
                job.status = JobStatus.RUNNING
                job.started_at = time.monotonic()
                job.log.append(
                    JobLogEntry(0, "Local worker", "Compilation started locally.", LogStatus.ACTIVE)
                )

            def stage(value: str) -> None:
                with self._lock:
                    current = self._jobs[job_id]
                    elapsed = time.monotonic() - (current.started_at or time.monotonic())
                    if current.stage != value:
                        current.completed_stages.append(current.stage)
                        active = next(
                            (
                                entry
                                for entry in reversed(current.log)
                                if entry.status is LogStatus.ACTIVE
                            ),
                            None,
                        )
                        if active is not None:
                            current.log.append(
                                JobLogEntry(
                                    elapsed,
                                    active.stage,
                                    "Stage completed.",
                                    LogStatus.COMPLETE,
                                    max(0, elapsed - active.elapsed_seconds),
                                )
                            )
                        current.stage = value
                        current.log.append(
                            JobLogEntry(
                                elapsed,
                                value,
                                "Processing local archive artifacts.",
                                LogStatus.ACTIVE,
                            )
                        )

            try:
                result = operation(stage)
            except Exception as exception:
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.error = f"Compilation failed safely ({type(exception).__name__})."
                    job.finished_at = time.monotonic()
                    job.log.append(
                        JobLogEntry(
                            job.finished_at - (job.started_at or job.finished_at),
                            job.stage,
                            job.error,
                            LogStatus.FAILED,
                        )
                    )
            else:
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = JobStatus.SUCCEEDED
                    job.result = result
                    job.finished_at = time.monotonic()
                    elapsed = job.finished_at - (job.started_at or job.finished_at)
                    job.log.append(
                        JobLogEntry(
                            elapsed,
                            job.stage,
                            "All artifacts rendered and inventoried.",
                            LogStatus.COMPLETE,
                        )
                    )

        self._executor.submit(execute)
        return job_id

    def get(self, job_id: str) -> JobSnapshot | None:
        """Return a detached snapshot for polling."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return JobSnapshot(
                job_id=job.job_id,
                status=job.status,
                stage=job.stage,
                started_at=job.started_at,
                finished_at=job.finished_at,
                result=job.result,
                error=job.error,
                completed_stages=list(job.completed_stages),
                log=list(job.log),
            )
