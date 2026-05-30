from __future__ import annotations

import math
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
    "easy": NpcConfig(difficulty="easy", cooldown_s=1.25, charge_s=1.25, attack_s=1.15),
    "normal": NpcConfig(difficulty="normal", cooldown_s=0.95, charge_s=1.0, attack_s=1.35),
    "hard": NpcConfig(difficulty="hard", cooldown_s=0.7, charge_s=0.8, attack_s=1.55),
}


class NpcOpponent:
    def __init__(self, config: NpcConfig | None = None) -> None:
        self.config = config or DIFFICULTY_CONFIGS["easy"]
        self._cycle_started_at = time.time()

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

    def state(
        self,
        frame_shape: tuple[int, int, int],
        target: BeamState | None,
        hp: int,
        battle_active: bool = True,
        starts_in: float = 0.0,
    ) -> BeamState:
        height, width = frame_shape[:2]
        chest = (int(width * 0.78), int(height * 0.56))
        radius = int(max(42, min(92, width * 0.07)))
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

        phase, ratio = self._phase()
        charging = phase == "charge"
        active = phase == "attack"
        powering = phase == "ready"
        confidence = 0.82 if active else 0.68 if charging else 0.55
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
            debug={
                "npc": True,
                "phase": phase,
                "phase_ratio": round(ratio, 3),
                "difficulty": self.config.difficulty,
                "target_detected": bool(target and target.detected),
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
        if state.active:
            right_hand = (state.origin[0] + 10, state.origin[1] + 12)
        elif state.charging:
            right_hand = (state.origin[0] + 22, state.origin[1] + 4)

        cv2.circle(frame, head, int(radius * 0.38), core, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, head, int(radius * 0.38), outline, 2, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_top, torso_bottom, outline, 5, lineType=cv2.LINE_AA)
        cv2.line(frame, (chest[0] - int(radius * 0.7), chest[1] - int(radius * 0.12)), left_hand, palette, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, (chest[0] + int(radius * 0.7), chest[1] - int(radius * 0.12)), right_hand, palette, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_bottom, (chest[0] - int(radius * 0.42), chest[1] + int(radius * 1.3)), outline, 4, lineType=cv2.LINE_AA)
        cv2.line(frame, torso_bottom, (chest[0] + int(radius * 0.42), chest[1] + int(radius * 1.3)), outline, 4, lineType=cv2.LINE_AA)

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

    def _phase(self) -> tuple[str, float]:
        elapsed = time.time() - self._cycle_started_at
        total = max(0.1, self.config.cooldown_s + self.config.charge_s + self.config.attack_s)
        t = elapsed % total
        if t < self.config.cooldown_s:
            return "ready", t / max(0.01, self.config.cooldown_s)
        t -= self.config.cooldown_s
        if t < self.config.charge_s:
            return "charge", t / max(0.01, self.config.charge_s)
        t -= self.config.charge_s
        return "attack", t / max(0.01, self.config.attack_s)

    @staticmethod
    def _direction(origin: tuple[int, int], target: tuple[int, int]) -> tuple[float, float]:
        dx = float(target[0] - origin[0])
        dy = float(target[1] - origin[1])
        norm = math.hypot(dx, dy)
        if norm <= 1e-5:
            return (-1.0, 0.0)
        return (dx / norm, dy / norm)
