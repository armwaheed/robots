"""Tests for the interactive Unitree G1 demo policy."""

import asyncio

from strands_robots.policies import create_policy


def test_g1_demo_policy_registered():
    policy = create_policy("g1_demo")
    policy.set_robot_state_keys(
        [
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
        ]
    )

    actions = asyncio.run(
        policy.get_actions(
            {
                "waist_yaw_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_elbow_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_elbow_joint": 0.0,
            },
            "pick up the cube with the right arm",
        )
    )

    assert actions
    assert "right_shoulder_pitch_joint" in actions[0]
    assert actions[0]["right_elbow_joint"] != 0.0
