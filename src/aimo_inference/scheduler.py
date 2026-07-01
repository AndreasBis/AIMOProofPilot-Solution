from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Protocol

from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.io import AIMOProblemResult
from aimo_inference.sandbox import AIMOSandbox


class AIMOProblemEngine(Protocol):

    def run_problem(self, record: AIMOProblemRecord) -> AIMOProblemResult:

        ...


@dataclass(frozen=True)
class AIMOSequenceSandbox:

    sequence_index: int
    sandbox: AIMOSandbox


@dataclass(frozen=True)
class AIMORolloutTopology:

    group_size: int
    active_problem_count: int
    sandbox_count: int

    @property
    def active_sequence_count(self) -> int:

        return self.group_size * self.active_problem_count


@dataclass(frozen=True)
class AIMOScheduleSummary:

    total_records: int
    sharded_records: int
    succeeded: int
    failed: int
    timed_out: int
    sequence_count: int
    group_size: int
    active_problem_count: int
    elapsed_seconds: float


class AIMOScheduler:

    def __init__(self, config: AIMOConfig, engine: AIMOProblemEngine) -> None:

        self.config = config
        self.engine = engine

    def shard_records(self, records: list[AIMOProblemRecord]) -> list[AIMOProblemRecord]:

        if self.config.world_size < 1:
            raise ValueError("world_size must be at least 1.")

        if not 0 <= self.config.global_rank < self.config.world_size:
            raise ValueError("global_rank must satisfy 0 <= global_rank < world_size.")

        return [
            record
            for record in records
            if record.order_index % self.config.world_size == self.config.global_rank
        ]

    def run(self, records: list[AIMOProblemRecord]) -> tuple[list[AIMOProblemResult], AIMOScheduleSummary]:

        started_at = time.monotonic()
        sharded_records = self.shard_records(records)
        max_workers = self._max_workers()

        if max_workers == 1:
            sequence_sandbox = AIMOSequenceSandbox(
                sequence_index=0,
                sandbox=AIMOSandbox(config=self.config),
            )
            results = [
                self._run_record(
                    record=record,
                    sequence_sandbox=sequence_sandbox,
                )
                for record in sharded_records
            ]
        else:
            results = self._run_concurrent(sharded_records, max_workers=max_workers)

        ordered_results = sorted(results, key=lambda result: result.order_index)
        timed_out = sum(
            1
            for result in ordered_results
            if result.metadata.get("timed_out") is True
        )
        succeeded = sum(
            1
            for result in ordered_results
            if result.success
        )
        failed = len(ordered_results) - succeeded
        summary = AIMOScheduleSummary(
            total_records=len(records),
            sharded_records=len(sharded_records),
            succeeded=succeeded,
            failed=failed,
            timed_out=timed_out,
            sequence_count=max_workers,
            group_size=self.config.group_size,
            active_problem_count=self.config.active_problem_count,
            elapsed_seconds=time.monotonic() - started_at,
        )

        return ordered_results, summary

    def _run_concurrent(
        self,
        records: list[AIMOProblemRecord],
        max_workers: int,
    ) -> list[AIMOProblemResult]:

        results: list[AIMOProblemResult] = []
        pending_records = iter(records)
        sequence_sandboxes = [
            AIMOSequenceSandbox(
                sequence_index=sequence_index,
                sandbox=AIMOSandbox(config=self.config),
            )
            for sequence_index in range(max_workers)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            running: dict[
                concurrent.futures.Future[AIMOProblemResult],
                tuple[AIMOProblemRecord, AIMOSequenceSandbox],
            ] = {}

            for sequence_sandbox in sequence_sandboxes:
                try:
                    record = next(pending_records)
                except StopIteration:
                    break

                running[executor.submit(
                    self._run_record,
                    record,
                    sequence_sandbox,
                )] = (record, sequence_sandbox)

            while running:
                done_futures, _ = concurrent.futures.wait(
                    running,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done_futures:
                    record, sequence_sandbox = running.pop(future)
                    results.append(self._result_from_future(future=future, record=record))

                    try:
                        next_record = next(pending_records)
                    except StopIteration:
                        continue

                    running[executor.submit(
                        self._run_record,
                        next_record,
                        sequence_sandbox,
                    )] = (next_record, sequence_sandbox)

        return results

    def _run_record(
        self,
        record: AIMOProblemRecord,
        sequence_sandbox: AIMOSequenceSandbox | None = None,
    ) -> AIMOProblemResult:

        started_at = time.monotonic()

        try:
            result = self._run_engine(
                record=record,
                sequence_sandbox=sequence_sandbox,
            )
        except Exception as error:
            return self._failure_result(
                record=record,
                error=str(error),
                timed_out=False,
                elapsed_seconds=time.monotonic() - started_at,
            )

        elapsed_seconds = time.monotonic() - started_at

        if elapsed_seconds > self.config.problem_timeout_seconds:
            return self._failure_result(
                record=record,
                error="Problem exceeded configured timeout.",
                timed_out=True,
                elapsed_seconds=elapsed_seconds,
            )

        return result

    def _run_engine(
        self,
        record: AIMOProblemRecord,
        sequence_sandbox: AIMOSequenceSandbox | None,
    ) -> AIMOProblemResult:

        if sequence_sandbox is not None and hasattr(self.engine, "run_problem_with_sandbox"):
            result = self.engine.run_problem_with_sandbox(
                record,
                sequence_sandbox.sandbox,
            )
        else:
            result = self.engine.run_problem(record)

        if sequence_sandbox is not None:
            result.metadata["sequence_index"] = sequence_sandbox.sequence_index

        return result

    def _result_from_future(
        self,
        future: concurrent.futures.Future[AIMOProblemResult],
        record: AIMOProblemRecord,
    ) -> AIMOProblemResult:

        try:
            return future.result()
        except Exception as error:
            return self._failure_result(
                record=record,
                error=str(error),
                timed_out=False,
                elapsed_seconds=0.0,
            )

    def _failure_result(
        self,
        record: AIMOProblemRecord,
        error: str,
        timed_out: bool,
        elapsed_seconds: float,
    ) -> AIMOProblemResult:

        return AIMOProblemResult(
            order_index=record.order_index,
            id=record.id,
            prediction="No proof was produced.",
            success=False,
            error=error,
            metadata={
                "timed_out": timed_out,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    def _max_workers(self) -> int:

        if self.config.mode in {"colab", "kaggle"}:
            return 1

        topology = AIMORolloutTopology(
            group_size=self.config.group_size,
            active_problem_count=self.config.active_problem_count,
            sandbox_count=self.config.sandbox_count,
        )

        return max(
            1,
            min(
                self.config.max_num_seqs,
                self.config.max_running_problems,
                topology.sandbox_count,
                topology.active_sequence_count,
            ),
        )
