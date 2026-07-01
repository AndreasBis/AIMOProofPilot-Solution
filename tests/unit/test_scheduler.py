from __future__ import annotations

from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.io import AIMOProblemResult
from aimo_inference.scheduler import AIMORolloutTopology
from aimo_inference.scheduler import AIMOScheduler
from aimo_training.queue import AIMOInterleavedGroupBuilder
from aimo_training.schema import AIMOTrainingRecord
from conftest import rollout_sample


class RecordingEngine:

    def __init__(self, fail_on: str = "") -> None:

        self.fail_on = fail_on
        self.records: list[str] = []

    def run_problem(self, record: AIMOProblemRecord) -> AIMOProblemResult:

        self.records.append(record.id)

        if record.id == self.fail_on:
            raise RuntimeError("engine failed")

        return AIMOProblemResult(
            order_index=record.order_index,
            id=record.id,
            prediction=f"Proof {record.id}",
            success=True,
            error="",
            metadata={},
        )

    def run_problem_with_sandbox(
        self,
        record: AIMOProblemRecord,
        sandbox: object,
    ) -> AIMOProblemResult:

        return self.run_problem(record)


def problem_records(count: int) -> list[AIMOProblemRecord]:

    return [
        AIMOProblemRecord(
            order_index=index,
            id=f"p{index}",
            problem=f"Problem {index}",
            metadata={},
        )
        for index in range(count)
    ]


def training_records(count: int) -> list[AIMOTrainingRecord]:

    return [
        AIMOTrainingRecord(
            order_index=index,
            id=f"p{index}",
            problem=f"Problem {index}",
            reference_solution=f"Reference {index}",
            metadata={},
        )
        for index in range(count)
    ]


def test_record_sharding_by_rank_and_world_size() -> None:

    scheduler = AIMOScheduler(
        config=AIMOConfig(
            global_rank=1,
            world_size=3,
        ),
        engine=RecordingEngine(),
    )

    sharded_records = scheduler.shard_records(problem_records(7))

    assert [record.order_index for record in sharded_records] == [
        1,
        4,
    ]


def test_invalid_rank_and_world_size_handling() -> None:

    records = problem_records(2)

    for config in [
        AIMOConfig(world_size=0),
        AIMOConfig(global_rank=2, world_size=2),
    ]:
        scheduler = AIMOScheduler(
            config=config,
            engine=RecordingEngine(),
        )

        try:
            scheduler.shard_records(records)
        except ValueError as error:
            assert "world_size" in str(error)
        else:
            raise AssertionError("Invalid scheduler topology was accepted.")


def test_sequential_execution_in_local_mode() -> None:

    engine = RecordingEngine()
    scheduler = AIMOScheduler(
        config=AIMOConfig(mode="colab"),
        engine=engine,
    )

    results, summary = scheduler.run(problem_records(3))

    assert [result.id for result in results] == [
        "p0",
        "p1",
        "p2",
    ]
    assert engine.records == [
        "p0",
        "p1",
        "p2",
    ]
    assert summary.sequence_count == 1
    assert summary.succeeded == 3
    assert summary.failed == 0


def test_concurrent_execution_in_singularity_mode() -> None:

    scheduler = AIMOScheduler(
        config=AIMOConfig(
            mode="singularity",
            max_num_seqs=2,
            max_running_problems=2,
            sandbox_count=96,
        ),
        engine=RecordingEngine(),
    )

    results, summary = scheduler.run(problem_records(4))

    assert [result.id for result in results] == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]
    assert summary.sequence_count == 2
    assert {
        result.metadata["sequence_index"]
        for result in results
    } == {
        0,
        1,
    }


def test_timeout_metadata() -> None:

    scheduler = AIMOScheduler(
        config=AIMOConfig(problem_timeout_seconds=-1.0),
        engine=RecordingEngine(),
    )

    results, summary = scheduler.run(problem_records(1))

    assert results[0].success is False
    assert results[0].metadata["timed_out"] is True
    assert summary.timed_out == 1


def test_exception_to_failure_result_conversion() -> None:

    scheduler = AIMOScheduler(
        config=AIMOConfig(),
        engine=RecordingEngine(fail_on="p1"),
    )

    results, summary = scheduler.run(problem_records(2))

    assert results[1].success is False
    assert results[1].prediction == "No proof was produced."
    assert "engine failed" in results[1].error
    assert summary.failed == 1


def test_interleaved_group_scheduling_admits_new_problem_after_completion() -> None:

    builder = AIMOInterleavedGroupBuilder(
        records=training_records(3),
        group_size=2,
        active_problem_count=2,
    )

    assert builder.active_problem_ids() == [
        "p0",
        "p1",
    ]
    assert builder.add_sample(rollout_sample(problem_id="p0", rollout_index=0)) is None
    completed_group = builder.add_sample(rollout_sample(problem_id="p0", rollout_index=1))

    assert completed_group is not None
    assert completed_group.problem_id == "p0"
    assert builder.active_problem_ids() == [
        "p1",
        "p2",
    ]


def test_partial_groups_are_not_emitted() -> None:

    builder = AIMOInterleavedGroupBuilder(
        records=training_records(1),
        group_size=3,
        active_problem_count=1,
    )

    assert builder.add_sample(rollout_sample(problem_id="p0", rollout_index=0)) is None
    assert builder.add_sample(rollout_sample(problem_id="p0", rollout_index=1)) is None


def test_six_active_problems_with_sixteen_rollouts_need_ninety_six_sandboxes() -> None:

    topology = AIMORolloutTopology(
        group_size=16,
        active_problem_count=6,
        sandbox_count=96,
    )
    scheduler = AIMOScheduler(
        config=AIMOConfig(
            mode="singularity",
            group_size=16,
            active_problem_count=6,
            sandbox_count=96,
            max_num_seqs=128,
            max_running_problems=128,
        ),
        engine=RecordingEngine(),
    )

    assert topology.active_sequence_count == 96
    assert scheduler._max_workers() == 96
