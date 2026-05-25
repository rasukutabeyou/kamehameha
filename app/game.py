from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.detector import KameState


@dataclass
class HitEvent:
    target_id: int
    center: tuple[int, int]
    damage: int
    ttl: int = 10
    age: int = 0


@dataclass
class GameSnapshot:
    hp: list[int]
    max_hp: int
    last_hit: int | None = None
    winner: int | None = None
    hits: int = 0


@dataclass
class KameGame:
    players: int
    max_hp: int = 100
    damage: int = 6
    hit_cooldown_s: float = 0.38
    hp: list[int] = field(default_factory=list)
    last_damage_at: list[float] = field(default_factory=list)
    hit_events: list[HitEvent] = field(default_factory=list)
    hits: int = 0
    last_hit: int | None = None

    def __post_init__(self) -> None:
        self.players = max(1, self.players)
        if not self.hp:
            self.hp = [self.max_hp for _ in range(self.players)]
        if not self.last_damage_at:
            self.last_damage_at = [0.0 for _ in range(self.players)]

    def reset(self) -> None:
        self.hp = [self.max_hp for _ in range(self.players)]
        self.last_damage_at = [0.0 for _ in range(self.players)]
        self.hit_events.clear()
        self.hits = 0
        self.last_hit = None

    def update(
        self,
        states: list[KameState],
        frame_shape: tuple[int, int, int],
        beam_collision: tuple[tuple[int, int], float] | None,
    ) -> None:
        now = time.time()
        state_by_id = {state.player_id: state for state in states}
        for target in states:
            if not target.detected or target.chest_radius <= 0 or self.hp[target.player_id] <= 0:
                continue
            if now - self.last_damage_at[target.player_id] < self.hit_cooldown_s:
                continue
            for shooter in states:
                if shooter.player_id == target.player_id or not shooter.active or shooter.mode != "beam":
                    continue
                if self._beam_hits_chest(shooter, target, frame_shape, beam_collision):
                    self.hp[target.player_id] = max(0, self.hp[target.player_id] - self.damage)
                    self.last_damage_at[target.player_id] = now
                    self.hit_events.append(HitEvent(target.player_id, target.chest_center, self.damage))
                    self.hits += 1
                    self.last_hit = target.player_id
                    break

        self.hit_events = [event for event in self.hit_events if event.age < event.ttl]
        _ = state_by_id

    def snapshot(self) -> GameSnapshot:
        alive = [index for index, hp in enumerate(self.hp) if hp > 0]
        winner = None
        if self.players >= 2 and len(alive) == 1:
            winner = alive[0]
        return GameSnapshot(
            hp=list(self.hp),
            max_hp=self.max_hp,
            last_hit=self.last_hit,
            winner=winner,
            hits=self.hits,
        )

    def draw_overlay(self, frame: np.ndarray, states: list[KameState]) -> np.ndarray:
        for state in states:
            if state.detected and state.chest_radius > 0:
                color = (70, 210, 255) if state.player_id == 0 else (245, 110, 230)
                cv2.circle(frame, state.chest_center, state.chest_radius, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, state.chest_center, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        self._draw_hp_bars(frame)
        next_events: list[HitEvent] = []
        for event in self.hit_events:
            progress = event.age / max(1, event.ttl)
            fade = max(0.0, 1.0 - progress)
            radius = int(24 + progress * 44)
            cv2.circle(frame, event.center, radius, (255, 255, 255), max(1, int(5 * fade)), lineType=cv2.LINE_AA)
            cv2.circle(frame, event.center, max(4, radius // 3), (40, 40, 255), -1, lineType=cv2.LINE_AA)
            label = f"-{event.damage}"
            cv2.putText(
                frame,
                label,
                (event.center[0] + 16, event.center[1] - 14 - event.age * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            event.age += 1
            if event.age < event.ttl:
                next_events.append(event)
        self.hit_events = next_events
        return frame

    def _draw_hp_bars(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        bar_w = min(330, max(180, width // 3))
        bar_h = 18
        y = 18
        positions = [(22, y), (width - bar_w - 22, y)]
        colors = [(70, 210, 255), (245, 110, 230)]
        for player_id in range(min(self.players, 2)):
            x, y0 = positions[player_id]
            hp_ratio = self.hp[player_id] / max(1, self.max_hp)
            cv2.rectangle(frame, (x - 3, y0 - 3), (x + bar_w + 3, y0 + bar_h + 3), (10, 12, 16), -1)
            cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h), (54, 61, 72), -1)
            cv2.rectangle(frame, (x, y0), (x + int(bar_w * hp_ratio), y0 + bar_h), colors[player_id], -1)
            cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h), (230, 235, 245), 1)
            text = f"P{player_id + 1} HP {self.hp[player_id]}"
            tx = x if player_id == 0 else x + bar_w - 104
            cv2.putText(frame, text, (tx, y0 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 247, 250), 2, cv2.LINE_AA)

        snapshot = self.snapshot()
        if snapshot.winner is not None:
            label = f"P{snapshot.winner + 1} WIN"
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
            cv2.putText(
                frame,
                label,
                ((width - size[0]) // 2, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

    def _beam_hits_chest(
        self,
        shooter: KameState,
        target: KameState,
        frame_shape: tuple[int, int, int],
        beam_collision: tuple[tuple[int, int], float] | None,
    ) -> bool:
        height, width = frame_shape[:2]
        origin = np.array(shooter.origin, dtype=np.float32)
        direction = np.array(shooter.direction, dtype=np.float32)
        chest = np.array(target.chest_center, dtype=np.float32)
        to_chest = chest - origin
        along = float(np.dot(to_chest, direction))
        if along <= 0.0:
            return False

        max_len = math.hypot(width, height) * 1.35
        if beam_collision is not None:
            collision = np.array(beam_collision[0], dtype=np.float32)
            collision_along = float(np.dot(collision - origin, direction))
            if collision_along > 0.0:
                max_len = min(max_len, collision_along)
        if along > max_len:
            return False

        closest = origin + direction * along
        distance = float(np.linalg.norm(chest - closest))
        beam_width = shooter.radius * (0.62 + shooter.confidence * 0.52)
        return distance <= target.chest_radius + beam_width
