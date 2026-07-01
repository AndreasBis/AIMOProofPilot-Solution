from __future__ import annotations

from aimo_training.queue import AIMOInterleavedGroupBuilder
from aimo_training.schema import AIMOTrainingRecord
from conftest import rollout_sample


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


def test_interleaved_grpo_queue_smoke() -> None:

    builder = AIMOInterleavedGroupBuilder(
        records=training_records(7),
        group_size=16,
        active_problem_count=6,
    )

    assert builder.active_problem_ids() == [
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
    ]

    completed_groups = []

    for rollout_index in range(15):
        assert builder.add_sample(
            rollout_sample(
                problem_id="p0",
                rollout_index=rollout_index,
            )
        ) is None

    assert builder.add_sample(
        rollout_sample(
            problem_id="p1",
            rollout_index=0,
        )
    ) is None
    completed_groups.append(
        builder.add_sample(
            rollout_sample(
                problem_id="p0",
                rollout_index=15,
            )
        )
    )

    assert completed_groups[0] is not None
    assert completed_groups[0].problem_id == "p0"
    assert len(completed_groups[0].samples) == 16
    assert builder.active_problem_ids() == [
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
    ]
