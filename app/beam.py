from __future__ import annotations

import time
from dataclasses import dataclass, replace

from app.config import GameConfig
from app.detector import KameState


@dataclass
class PlayerBeamState:
    charge_started_at: float | None = None
    shot_started_at: float | None = None
    exhausted: bool = False


class BeamController:
    def __init__(self, players: int, config: GameConfig) -> None:
        self._config = config
        self._players = max(1, players)
        self._states = [PlayerBeamState() for _ in range(self._players)]

    def reset(self) -> None:
        self._states = [PlayerBeamState() for _ in range(self._players)]

    def apply(self, detected_states: list[KameState], ki: list[float] | None = None) -> list[KameState]:
        now = time.time()
        return [self._apply_player(state, now, ki) for state in detected_states]

    def _apply_player(self, state: KameState, now: float, ki: list[float] | None = None) -> KameState:
        if state.player_id >= len(self._states):
            return replace(state, active=False, charging=False, charge_ratio=0.0, beam_ratio=0.0)

        beam = self._states[state.player_id]
        pose_engaged = state.detected and (state.charging or state.active)
        if not pose_engaged:
            beam.charge_started_at = None
            beam.shot_started_at = None
            beam.exhausted = False
            return replace(state, active=False, charging=False, charge_ratio=0.0, beam_ratio=0.0)

        has_enough_ki = (
            ki is None
            or (state.player_id < len(ki) and ki[state.player_id] >= self._config.beam_ki_cost)
        )
        if beam.shot_started_at is None and not has_enough_ki:
            beam.charge_started_at = None
            beam.exhausted = False
            return replace(state, active=False, charging=False, charge_ratio=0.0, beam_ratio=0.0)

        if beam.charge_started_at is None:
            beam.charge_started_at = now

        charge_s = max(0.0, self._config.beam_charge_s)
        duration_s = max(0.01, self._config.beam_duration_s)
        charge_elapsed = now - beam.charge_started_at
        charge_ratio = 1.0 if charge_s <= 0.0 else min(1.0, charge_elapsed / charge_s)

        if beam.shot_started_at is not None:
            shot_elapsed = now - beam.shot_started_at
            beam_ratio = max(0.0, 1.0 - shot_elapsed / duration_s)
            if shot_elapsed >= duration_s:
                beam.exhausted = True
                beam.shot_started_at = None
                return replace(state, active=False, charging=False, charge_ratio=1.0, beam_ratio=0.0)
            return replace(state, active=state.active, charging=False, charge_ratio=1.0, beam_ratio=beam_ratio)

        if beam.exhausted:
            return replace(state, active=False, charging=False, charge_ratio=1.0, beam_ratio=0.0)

        can_fire = charge_ratio >= 1.0 and state.active
        if not can_fire:
            return replace(state, active=False, charging=True, charge_ratio=charge_ratio, beam_ratio=0.0)

        if ki is not None:
            ki[state.player_id] = max(0.0, ki[state.player_id] - self._config.beam_ki_cost)

        beam.shot_started_at = now
        return replace(state, active=True, charging=False, charge_ratio=1.0, beam_ratio=1.0)
