from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.detector import BeamState


@dataclass(frozen=True)
class NpcConfig:
    player_id: int = 1
    cooldown_s: float = 1.25
    charge_s: float = 1.25
    attack_s: float = 1.15
    difficulty: str = "easy"


DIFFICULTY_CONFIGS: dict[str, NpcConfig] = {
    "easy": NpcConfig(difficulty="easy", cooldown_s=2.2, charge_s=1.15, attack_s=1.05),
    "normal": NpcConfig(difficulty="normal", cooldown_s=1.45, charge_s=1.25, attack_s=2.55),
    "hard": NpcConfig(difficulty="hard", cooldown_s=1.1, charge_s=1.15, attack_s=2.75),
}


class NpcOpponent:
    def __init__(self, config: NpcConfig | None = None) -> None:
        self.config = config or DIFFICULTY_CONFIGS["easy"]
        self._cycle_started_at = time.time()
        self._rng = random.Random(2026)
        self._tactic: str | None = None
        self._tactic_started_at = 0.0
        self._tactic_until = 0.0
        self._last_reaction_at = 0.0
        self._recovering_energy = False
        self._evade_sign = 1

    @property
    def difficulty(self) -> str:
        return self.config.difficulty

    @property
    def starts_ultra(self) -> bool:
        return self.config.difficulty == "hard"

    def set_difficulty(self, difficulty: str) -> str:
        normalized = difficulty.lower().strip()
        self.config = DIFFICULTY_CONFIGS.get(normalized, DIFFICULTY_CONFIGS["easy"])
        self.reset()
        return self.config.difficulty

    def reset(self) -> None:
        self._cycle_started_at = time.time()
        self._tactic = None
        self._tactic_started_at = 0.0
        self._tactic_until = 0.0
        self._last_reaction_at = 0.0
        self._recovering_energy = False

    def state(
        self,
        frame_shape: tuple[int, int, int],
        target: BeamState | None,
        hp: int,
        energy: float | None = None,
        beam_energy_cost: float = 25.0,
        beam_charge_s: float = 1.15,
        ultra: bool = False,
        ultra_energy_drain_per_s: float = 0.0,
        battle_active: bool = True,
        starts_in: float = 0.0,
    ) -> BeamState:
        height, width = frame_shape[:2]
        now = time.time()
        radius = int(max(42, min(92, width * 0.07)))
        base_chest = (int(width * 0.78), int(height * 0.56))
        phase_hint = self._select_tactic(target, hp, now, energy, beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s) if battle_active else None
        chest = self._chest_position(base_chest, radius, now, phase_hint)
        origin = (chest[0] - int(radius * 0.95), chest[1] - int(radius * 0.05))

        if target and target.detected and target.chest_radius > 0:
            target_point = target.chest_center
        else:
            target_point = (int(width * 0.24), int(height * 0.54))

        direction = self._direction(origin, target_point)
        if hp <= 0:
            return BeamState(
                active=False,
                charging=False,
                confidence=0.0,
                origin=origin,
                direction=direction,
                radius=radius,
                mode="idle",
                player_id=self.config.player_id,
                detected=False,
                chest_center=chest,
                chest_radius=0,
                debug={"npc": True, "phase": "down", "difficulty": self.config.difficulty},
            )

        if not battle_active:
            return BeamState(
                active=False,
                charging=False,
                confidence=0.45,
                origin=origin,
                direction=direction,
                radius=radius,
                mode="idle",
                player_id=self.config.player_id,
                detected=True,
                chest_center=chest,
                chest_radius=0,
                debug={
                    "npc": True,
                    "phase": "waiting",
                    "starts_in": round(max(0.0, starts_in), 1),
                    "difficulty": self.config.difficulty,
                    "target_detected": bool(target and target.detected),
                    "direction": (round(direction[0], 3), round(direction[1], 3)),
                },
            )

        phase, ratio = self._phase(now, energy, beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s)
        if phase_hint is not None:
            phase, ratio = phase_hint
        charging = phase in {"charge", "counter_charge"}
        active = phase in {"attack", "counter_attack"}
        powering = phase in {"ready", "recover"}
        confidence = 0.88 if phase == "counter_attack" else 0.82 if active else 0.7 if charging else 0.6
        return BeamState(
            active=active,
            charging=charging,
            confidence=confidence,
            origin=origin,
            direction=direction,
            radius=radius,
            mode="beam",
            player_id=self.config.player_id,
            detected=True,
            chest_center=chest,
            chest_radius=radius,
            powering=powering,
            guarding=phase == "guard",
            debug={
                "npc": True,
                "phase": phase,
                "phase_ratio": round(ratio, 3),
                "difficulty": self.config.difficulty,
                "target_detected": bool(target and target.detected),
                "target_active": bool(target and target.active),
                "target_charging": bool(target and target.charging),
                "target_powering": bool(target and target.powering),
                "energy": round(energy, 1) if energy is not None else None,
                "beam_energy_cost": round(beam_energy_cost, 1),
                "required_attack_energy": round(self._required_attack_energy(beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s), 1),
                "ultra": ultra,
                "ultra_energy_drain_per_s": round(ultra_energy_drain_per_s, 2),
                "direction": (round(direction[0], 3), round(direction[1], 3)),
            },
        )

    def draw(self, frame: np.ndarray, state: BeamState) -> None:
        if not state.detected and state.chest_radius <= 0:
            return
        chest = state.chest_center
        radius = max(36, state.chest_radius)
        palette = (245, 110, 230)
        core = (90, 35, 110)
        outline = (255, 205, 255)

        head = (chest[0], chest[1] - int(radius * 1.2))
        torso_top = (chest[0], chest[1] - int(radius * 0.55))
        torso_bottom = (chest[0], chest[1] + int(radius * 0.75))
        left_hand = state.origin
        right_hand = (chest[0] + int(radius * 0.55), chest[1] + int(radius * 0.08))
        phase = str(state.debug.get("phase", "")) if state.debug else ""
        if state.active:
            right_hand = (state.origin[0] + 10, state.origin[1] + 12)
        elif state.charging:
            right_hand = (state.origin[0] + 22, state.origin[1] + 4)
        elif phase == "guard":
            left_hand = (chest[0] - int(radius * 0.45), chest[1] - int(radius * 0.35))
            right_hand = (chest[0] - int(radius * 0.2), chest[1] + int(radius * 0.28))

        cv2.circle(frame, head, int(radius * 0.38), core, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, head, int(radius * 0.38), outline, 2, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_top, torso_bottom, outline, 5, lineType=cv2.LINE_AA)
        cv2.line(frame, (chest[0] - int(radius * 0.7), chest[1] - int(radius * 0.12)), left_hand, palette, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, (chest[0] + int(radius * 0.7), chest[1] - int(radius * 0.12)), right_hand, palette, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_bottom, (chest[0] - int(radius * 0.42), chest[1] + int(radius * 1.3)), outline, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_bottom, (chest[0] + int(radius * 0.42), chest[1] + int(radius * 1.3)), outline, 4, lineType=cv2.LINE_AA)
        if phase == "evade":
            cv2.putText(frame, "EVADE", (chest[0] - int(radius * 0.65), chest[1] + int(radius * 1.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 235, 255), 2, cv2.LINE_AA)

        label = "NPC" if not state.ultra else "NPC ULTRA"
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(
            frame,
            label,
            (chest[0] - size[0] // 2, head[1] - int(radius * 0.58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 235, 255),
            2,
            cv2.LINE_AA,
        )

    def _phase(
        self,
        now: float,
        energy: float | None,
        beam_energy_cost: float,
        beam_charge_s: float,
        ultra: bool,
        ultra_energy_drain_per_s: float,
    ) -> tuple[str, float]:
        elapsed = now - self._cycle_started_at
        total = max(0.1, self.config.cooldown_s + self.config.charge_s + self.config.attack_s)
        t = elapsed % total
        if t < self.config.cooldown_s:
            scheduled = ("ready", t / max(0.01, self.config.cooldown_s))
        else:
            t -= self.config.cooldown_s
            if t < self.config.charge_s:
                scheduled = ("charge", t / max(0.01, self.config.charge_s))
            else:
                t -= self.config.charge_s
                scheduled = ("attack", t / max(0.01, self.config.attack_s))

        if scheduled[0] == "attack":
            return scheduled
        if self._should_recover_energy(energy, beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s):
            self._recovering_energy = True
            return "recover", 0.0
        if self._recovering_energy:
            self._recovering_energy = False
            self._cycle_started_at = now
            return "ready", 0.0
        return scheduled

    def _select_tactic(
        self,
        target: BeamState | None,
        hp: int,
        now: float,
        energy: float | None,
        beam_energy_cost: float,
        beam_charge_s: float,
        ultra: bool,
        ultra_energy_drain_per_s: float,
    ) -> tuple[str, float] | None:
        if self.config.difficulty == "easy" or target is None or not target.detected:
            self._expire_tactic(now)
            return None

        if self._tactic is not None and now < self._tactic_until:
            current = self._current_tactic(now)
            if current is not None and current[0] in {"counter_attack", "attack"}:
                return current
            if self._should_recover_energy(energy, beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s):
                self._expire_tactic(now)
                return None
            return current
        self._expire_tactic(now)

        if self._should_recover_energy(energy, beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s):
            return None

        if now - self._last_reaction_at < self._reaction_gap():
            return None

        target_active = bool(target.active)
        target_charging = bool(target.charging)
        target_powering = bool(target.powering)
        low_hp = hp <= 35
        roll = self._rng.random()

        if target_active:
            if self.config.difficulty == "hard":
                tactic = "evade" if roll < 0.55 else "guard" if roll < 0.82 else "counter"
            else:
                tactic = "guard" if roll < 0.55 else "evade" if roll < 0.82 else None
            return self._start_tactic(tactic, now)

        if target_charging:
            if self.config.difficulty == "hard":
                tactic = "counter" if roll < 0.65 else "guard" if roll < 0.82 else None
            else:
                tactic = "counter" if roll < 0.38 else "guard" if roll < 0.58 else None
            return self._start_tactic(tactic, now)

        if target_powering or low_hp:
            chance = 0.75 if self.config.difficulty == "hard" else 0.42
            if roll < chance:
                return self._start_tactic("counter", now)

        return None

    def _start_tactic(self, tactic: str | None, now: float) -> tuple[str, float] | None:
        if tactic is None:
            return None
        durations = {
            "guard": 0.55 if self.config.difficulty == "hard" else 0.7,
            "evade": 0.62 if self.config.difficulty == "hard" else 0.78,
            "counter": 3.45 if self.config.difficulty == "hard" else 3.65,
        }
        self._tactic = tactic
        self._tactic_started_at = now
        self._tactic_until = now + durations[tactic]
        self._last_reaction_at = now
        if tactic == "evade":
            self._evade_sign *= -1
        return self._current_tactic(now)

    def _current_tactic(self, now: float) -> tuple[str, float] | None:
        if self._tactic is None:
            return None
        duration = max(0.01, self._tactic_until - self._tactic_started_at)
        ratio = max(0.0, min(1.0, (now - self._tactic_started_at) / duration))
        if self._tactic == "counter":
            charge_portion = 0.36 if self.config.difficulty == "hard" else 0.34
            if ratio < charge_portion:
                return "counter_charge", ratio / charge_portion
            return "counter_attack", (ratio - charge_portion) / max(0.01, 1.0 - charge_portion)
        return self._tactic, ratio

    def _expire_tactic(self, now: float) -> None:
        if self._tactic is not None and now >= self._tactic_until:
            self._tactic = None

    def _reaction_gap(self) -> float:
        return 0.55 if self.config.difficulty == "hard" else 1.0

    def _should_recover_energy(
        self,
        energy: float | None,
        beam_energy_cost: float,
        beam_charge_s: float,
        ultra: bool,
        ultra_energy_drain_per_s: float,
    ) -> bool:
        if energy is None:
            return False
        return energy < self._required_attack_energy(beam_energy_cost, beam_charge_s, ultra, ultra_energy_drain_per_s)

    def _required_attack_energy(
        self,
        beam_energy_cost: float,
        beam_charge_s: float,
        ultra: bool,
        ultra_energy_drain_per_s: float,
    ) -> float:
        minimum = max(beam_energy_cost, 1.0)
        margin = 8.0 if self.config.difficulty == "easy" else 12.0 if self.config.difficulty == "normal" else 16.0
        if ultra:
            charge_window_s = max(beam_charge_s, self.config.charge_s, 1.4)
            margin += max(0.0, ultra_energy_drain_per_s) * charge_window_s
        return minimum + margin

    def _chest_position(
        self,
        base_chest: tuple[int, int],
        radius: int,
        now: float,
        phase_hint: tuple[str, float] | None,
    ) -> tuple[int, int]:
        x, y = base_chest
        if self.config.difficulty in {"normal", "hard"}:
            x += int(math.sin(now * 1.2) * radius * (0.08 if self.config.difficulty == "normal" else 0.14))
        if phase_hint is not None and phase_hint[0] == "evade":
            ratio = phase_hint[1]
            arc = math.sin(math.pi * ratio)
            x += int(radius * 0.9 * arc)
            y += int(radius * 1.15 * arc * self._evade_sign)
        return (x, y)

    @staticmethod
    def _direction(origin: tuple[int, int], target: tuple[int, int]) -> tuple[float, float]:
        dx = float(target[0] - origin[0])
        dy = float(target[1] - origin[1])
        norm = math.hypot(dx, dy)
        if norm <= 1e-5:
            return (-1.0, 0.0)
        return (dx / norm, dy / norm)
