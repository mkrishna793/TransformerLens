"""Integration check for the consolidated Milestone-one/two runner."""

from research.cmae.run_dataset_safety_program import run_control_plane_demonstration


def test_control_plane_demonstration_replays_after_restart() -> None:
    result = run_control_plane_demonstration()
    assert result["committed"]
    assert result["restart_replay_equal"]
    assert result["calibration_count"] == 1
    assert result["replay_commits"] == 1
