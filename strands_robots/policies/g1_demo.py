"""Template-based natural-language demo policy for Unitree G1.

This is not a learned VLA. It exists to make the MuJoCo G1 demo interactive
through the same policy abstraction layer used by real providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from strands_robots.policies import Policy


@dataclass(frozen=True)
class _Phase:
    name: str
    steps: int


class G1DemoPolicy(Policy):
    """Instruction-conditioned pose sequencer for a simulated Unitree G1."""

    def __init__(self, default_side: str = "right", **kwargs):
        self.robot_state_keys: List[str] = []
        self.default_side = default_side
        self._active_instruction = ""
        self._plan: List[Dict[str, float]] = []
        self._cursor = 0

    @property
    def provider_name(self) -> str:
        return "g1_demo"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = list(robot_state_keys)

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        normalized = " ".join(instruction.lower().strip().split())
        if normalized != self._active_instruction or self._cursor >= len(self._plan):
            self._active_instruction = normalized
            self._plan = self._build_plan(observation_dict, normalized)
            self._cursor = 0

        if not self._plan:
            return [self._current_joint_state(observation_dict)]

        start = self._cursor
        end = min(len(self._plan), start + 8)
        self._cursor = end
        return self._plan[start:end]

    def _build_plan(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, float]]:
        current = self._current_joint_state(observation_dict)
        phases = self._select_phases(instruction)
        targets = [self._target_pose(current.keys(), phase, instruction) for phase in phases]

        plan: List[Dict[str, float]] = []
        previous = current
        for phase, target in zip(phases, targets):
            plan.extend(self._interpolate(previous, target, phase.steps))
            previous = target
        return plan

    def _select_phases(self, instruction: str) -> Sequence[_Phase]:
        if not instruction:
            return [_Phase("home", 20)]

        if any(word in instruction for word in ("pick", "grab", "grasp", "cube", "object")):
            return [_Phase("reach", 24), _Phase("grasp", 20), _Phase("lift", 24), _Phase("retract", 20)]
        if any(word in instruction for word in ("reach", "touch", "point")):
            return [_Phase("reach", 24)]
        if any(word in instruction for word in ("lift", "raise")):
            return [_Phase("lift", 24)]
        if any(word in instruction for word in ("wave", "hello", "greet")):
            return [_Phase("wave_out", 18), _Phase("wave_in", 18), _Phase("wave_out", 18), _Phase("home", 18)]
        if "left" in instruction and any(word in instruction for word in ("turn", "rotate", "twist")):
            return [_Phase("turn_left", 18), _Phase("home", 18)]
        if "right" in instruction and any(word in instruction for word in ("turn", "rotate", "twist")):
            return [_Phase("turn_right", 18), _Phase("home", 18)]
        if any(word in instruction for word in ("open hand", "release", "drop")):
            return [_Phase("open", 18)]
        if any(word in instruction for word in ("home", "neutral", "reset", "stand")):
            return [_Phase("home", 20)]
        return [_Phase("reach", 20), _Phase("home", 20)]

    def _current_joint_state(self, observation_dict: Dict[str, Any]) -> Dict[str, float]:
        state = {}
        for key in self.robot_state_keys:
            value = observation_dict.get(key)
            if isinstance(value, (int, float)):
                state[key] = float(value)
        return state

    def _target_pose(
        self,
        joint_names: Iterable[str],
        phase: _Phase,
        instruction: str,
    ) -> Dict[str, float]:
        side = self._instruction_side(instruction)
        pose = {}
        for joint_name in joint_names:
            lower = joint_name.lower()
            if "floating_base" in lower:
                continue

            arm_sign = self._arm_sign(lower, side)

            if any(token in lower for token in ("hip", "knee", "ankle")):
                pose[joint_name] = 0.0
                continue

            if "waist_yaw" in lower:
                if phase.name == "turn_left":
                    pose[joint_name] = 0.45
                elif phase.name == "turn_right":
                    pose[joint_name] = -0.45
                elif phase.name == "reach":
                    pose[joint_name] = 0.10 * arm_sign
                else:
                    pose[joint_name] = 0.0
                continue

            if "waist_roll" in lower:
                pose[joint_name] = 0.0
                continue

            if "waist_pitch" in lower:
                pose[joint_name] = 0.10 if phase.name in ("reach", "grasp", "lift") else 0.0
                continue

            if "shoulder_pitch" in lower:
                pose[joint_name] = {
                    "reach": -0.55,
                    "grasp": -0.60,
                    "lift": -0.30,
                    "retract": -0.10,
                    "wave_out": -0.15 if arm_sign else -0.05,
                    "wave_in": -0.35 if arm_sign else -0.05,
                    "home": -0.05,
                    "open": -0.35,
                    "turn_left": -0.05,
                    "turn_right": -0.05,
                }.get(phase.name, -0.05)
                continue

            if "shoulder_roll" in lower:
                base = {
                    "reach": 0.28,
                    "grasp": 0.32,
                    "lift": 0.18,
                    "retract": 0.08,
                    "wave_out": 0.65,
                    "wave_in": 0.45,
                    "home": 0.02,
                    "open": 0.22,
                    "turn_left": 0.02,
                    "turn_right": 0.02,
                }.get(phase.name, 0.02)
                pose[joint_name] = base * arm_sign
                continue

            if "shoulder_yaw" in lower:
                base = {
                    "reach": 0.24,
                    "grasp": 0.28,
                    "lift": 0.12,
                    "retract": 0.04,
                    "wave_out": 0.18,
                    "wave_in": -0.18,
                    "home": 0.0,
                    "open": 0.08,
                }.get(phase.name, 0.0)
                pose[joint_name] = base * arm_sign
                continue

            if "elbow" in lower:
                pose[joint_name] = {
                    "reach": 0.90,
                    "grasp": 1.10,
                    "lift": 0.72,
                    "retract": 0.35,
                    "wave_out": 1.20,
                    "wave_in": 0.75,
                    "home": 0.18,
                    "open": 0.95,
                }.get(phase.name, 0.18)
                continue

            if "wrist_roll" in lower:
                pose[joint_name] = 0.08 * arm_sign if phase.name in ("reach", "grasp", "open") else 0.0
                continue

            if "wrist_pitch" in lower:
                pose[joint_name] = {
                    "reach": -0.12,
                    "grasp": -0.18,
                    "lift": -0.06,
                    "retract": 0.0,
                    "wave_out": 0.0,
                    "wave_in": 0.0,
                    "home": 0.0,
                    "open": -0.10,
                }.get(phase.name, 0.0)
                continue

            if "wrist_yaw" in lower:
                pose[joint_name] = 0.12 * arm_sign if phase.name in ("reach", "grasp", "open") else 0.0
                continue

            if "hand" in lower or "thumb" in lower or "index" in lower or "middle" in lower:
                pose[joint_name] = 0.55 if phase.name == "grasp" else 0.12 if phase.name == "lift" else 0.0
                continue

            pose[joint_name] = 0.0

        return pose

    def _interpolate(self, start: Dict[str, float], target: Dict[str, float], steps: int) -> List[Dict[str, float]]:
        keys = sorted(set(start) | set(target))
        sequence: List[Dict[str, float]] = []
        for step in range(steps):
            alpha = (step + 1) / max(steps, 1)
            sequence.append(
                {
                    key: start.get(key, 0.0) + (target.get(key, 0.0) - start.get(key, 0.0)) * alpha
                    for key in keys
                }
            )
        return sequence

    def _instruction_side(self, instruction: str) -> str:
        if "left" in instruction:
            return "left"
        if "right" in instruction:
            return "right"
        return self.default_side

    @staticmethod
    def _arm_sign(joint_name: str, active_side: str) -> int:
        if "left_" in joint_name:
            return 1 if active_side == "left" else 0
        if "right_" in joint_name:
            return -1 if active_side == "right" else 0
        return 1
