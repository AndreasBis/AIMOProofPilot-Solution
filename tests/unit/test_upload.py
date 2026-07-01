from __future__ import annotations

from execution.upload import result_inventory


def test_upload_inventory_detects_failure_reports_vllm_logs_and_preflight() -> None:

    inventory = result_inventory([
        {
            "path": "failure_artifacts/failure_report.json",
            "size_bytes": 10,
        },
        {
            "path": "failure_artifacts/online_servers/rank_0_judge/vllm_stderr.log",
            "size_bytes": 20,
        },
        {
            "path": "failure_artifacts/online_servers/rank_0_judge/vllm_command.json",
            "size_bytes": 30,
        },
        {
            "path": "failure_artifacts/online_servers/rank_0_judge/service_preflight.json",
            "size_bytes": 40,
        },
    ])

    assert inventory["has_failure_report"] is True
    assert inventory["has_vllm_diagnostics"] is True
    assert inventory["has_service_preflight_diagnostics"] is True
    assert inventory["failure_reports"] == [
        "failure_artifacts/failure_report.json",
    ]
