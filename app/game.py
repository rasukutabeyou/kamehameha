from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import GameConfig
from app.detector import BeamState


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
    energy: list[float]
    max_energy: float
    ultra: list[bool]
    damage: int
    hit_cooldown_s: float
    beam_charge_s: float
    beam_duration_s: float
    beam_energy_cost: float
    energy_charge_per_s: float
    energy_charge_damage_bonus: int
    guard_damage_multiplier: float
    ultra_energy_cost: float
    ultra_damage_multiplier: float
    ultra_energy_charge_multiplier: float
    ultra_damage_reduction: int
    ultra_energy_drain_per_s: float
    last_hit: int | None = None
    winner: int | None = None
    hits: int = 0


@dataclass
class BeamGame:
    players: int
    config: GameConfig = field(default_factory=GameConfig)
    hp: list[int] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    ultra: list[bool] = field(default_factory=list)
    last_damage_at: list[float] = field(default_factory=list)
    hit_events: list[HitEvent] = field(default_factory=list)
    hits: int = 0
    last_hit: int | None = None
    last_update_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.players = max(1, self.players)
        if not self.hp:
            self.hp = [self.config.max_hp for _ in range(self.players)]
        if not self.energy:
            initial_energy = min(self.config.energy_max, max(0.0, self.config.energy_initial))
            self.energy = [initial_energy for _ in range(self.players)]
        if not self.ultra:
            self.ultra = [False for _ in range(self.players)]
        if not self.last_damage_at:
            self.last_damage_at = [0.0 for _ in range(self.players)]

    def reset(self) -> None:
        self.hp = [self.config.max_hp for _ in range(self.players)]
        initial_energy = min(self.config.energy_max, max(0.0, self.config.energy_initial))
        self.energy = [initial_energy for _ in range(self.players)]
        self.ultra = [False for _ in range(self.players)]
        self.last_damage_at = [0.0 for _ in range(self.players)]
        self.hit_events.clear()
        self.hits = 0
        self.last_hit = None
        self.last_update_at = time.time()

    def update(
        self,
        states: list[BeamState],
        frame_shape: tuple[int, int, int],
        beam_collision: tuple[tuple[int, int], float] | None,
    ) -> None:
        now = time.time()
        self._update_energy_and_ultra(states, now)

        for target in states:
            if not target.detected or target.chest_radius <= 0 or self.hp[target.player_id] <= 0:
                continue
            if now - self.last_damage_at[target.player_id] < self.config.hit_cooldown_s:
                continue
            for shooter in states:
                if shooter.player_id == target.player_id or not shooter.active or shooter.mode != "beam":
                    continue
                if self._beam_hits_chest(shooter, target, frame_shape, beam_collision):
                    damage = self._attack_damage(shooter, target)
                    self.hp[target.player_id] = max(0, self.hp[target.player_id] - damage)
                    self.last_damage_at[target.player_id] = now
                    self.hit_events.append(HitEvent(target.player_id, target.chest_center, damage))
                    self.hits += 1
                    self.last_hit = target.player_id
                    break

        self.hit_events = [event for event in self.hit_events if event.age < event.ttl]

    def _update_energy_and_ultra(self, states: list[BeamState], now: float) -> None:
        dt = max(0.0, min(1.0, now - self.last_update_at))
        self.last_update_at = now
        for state in states:
            if state.player_id >= len(self.energy):
                continue
            player_id = state.player_id
            if self.hp[player_id] <= 0:
                self.ultra[player_id] = False
                state.ultra = False
                continue

            if self.ultra[player_id] and dt > 0.0:
                self.energy[player_id] = max(0.0, self.energy[player_id] - self.config.ultra_energy_drain_per_s * dt)
                if self.energy[player_id] <= 0.0:
                    self.ultra[player_id] = False

            if (
                state.detected
                and state.transforming
                and not self.ultra[player_id]
                and self.energy[player_id] >= self.config.ultra_energy_cost
            ):
                self.energy[player_id] = max(0.0, self.energy[player_id] - self.config.ultra_energy_cost)
                self.ultra[player_id] = True

            if state.detected and state.powering and dt > 0.0:
                charge_rate = self.config.energy_charge_per_s
                if self.ultra[player_id]:
                    charge_rate *= self.config.ultra_energy_charge_multiplier
                self.energy[player_id] = min(self.config.energy_max, self.energy[player_id] + charge_rate * dt)

            if self.ultra[player_id] and self.energy[player_id] <= 0.0:
                self.ultra[player_id] = False
            state.ultra = self.ultra[player_id]

    def _attack_damage(self, shooter: BeamState, target: BeamState) -> int:
        damage = float(self.config.damage)
        if self._is_ultra(shooter.player_id):
            damage *= self.config.ultra_damage_multiplier
        if target.powering:
            damage += self.config.energy_charge_damage_bonus
        if target.guarding:
            damage *= self.config.guard_damage_multiplier
        if self._is_ultra(target.player_id):
            damage -= self.config.ultra_damage_reduction
        return max(0, int(round(damage)))

    def _is_ultra(self, player_id: int) -> bool:
        return 0 <= player_id < len(self.ultra) and self.ultra[player_id]

    def snapshot(self) -> GameSnapshot:
        alive = [index for index, hp in enumerate(self.hp) if hp > 0]
        winner = None
        if self.players >= 2 and len(alive) == 1:
            winner = alive[0]
        return GameSnapshot(
            hp=list(self.hp),
            max_hp=self.config.max_hp,
            energy=[round(value, 1) for value in self.energy],
            max_energy=self.config.energy_max,
            ultra=list(self.ultra),
            damage=self.config.damage,
            hit_cooldown_s=self.config.hit_cooldown_s,
            beam_charge_s=self.config.beam_charge_s,
            beam_duration_s=self.config.beam_duration_s,
            beam_energy_cost=self.config.beam_energy_cost,
            energy_charge_per_s=self.config.energy_charge_per_s,
            energy_charge_damage_bonus=self.config.energy_charge_damage_bonus,
            guard_damage_multiplier=self.config.guard_damage_multiplier,
            ultra_energy_cost=self.config.ultra_energy_cost,
            ultra_damage_multiplier=self.config.ultra_damage_multiplier,
            ultra_energy_charge_multiplier=self.config.ultra_energy_charge_multiplier,
            ultra_damage_reduction=self.config.ultra_damage_reduction,
            ultra_energy_drain_per_s=self.config.ultra_energy_drain_per_s,
            last_hit=self.last_hit,
            winner=winner,
            hits=self.hits,
        )

    def draw_overlay(self, frame: np.ndarray, states: list[BeamState]) -> np.ndarray:
        self._draw_target_zones(frame, states)
        self._draw_player_bars(frame, states)
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

    def _draw_target_zones(self, frame: np.ndarray, states: list[BeamState]) -> None:
        for state in states:
            if not state.detected or state.chest_radius <= 0:
                continue
            color = (70, 210, 255) if state.player_id == 0 else (245, 110, 230)
            if state.ultra:
                color = (60, 245, 255)
            elif state.powering:
                color = (80, 255, 145)
            if state.guarding:
                x, y = state.chest_center
                r = state.chest_radius
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), color, 2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame, state.chest_center, state.chest_radius, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, state.chest_center, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def _draw_player_bars(self, frame: np.ndarray, states: list[BeamState]) -> None:
        height, width = frame.shape[:2]
        bar_w = min(330, max(180, width // 3))
        bar_h = 16
        y = 18
        positions = [(22, y), (width - bar_w - 22, y)]
        colors = [(70, 210, 255), (245, 110, 230)]
        states_by_id = {state.player_id: state for state in states}
        for player_id in range(min(self.players, 2)):
            x, y0 = positions[player_id]
            align_right = player_id == 1
            self._draw_bar(frame, x, y0, bar_w, bar_h, self.hp[player_id] / max(1, self.config.max_hp), colors[player_id], "HP")
            self._draw_bar(frame, x, y0 + 24, bar_w, bar_h, self.energy[player_id] / max(1.0, self.config.energy_max), (88, 230, 120), "ENERGY")
            state = states_by_id.get(player_id)
            beam_ratio = state.beam_ratio if state else 0.0
            self._draw_bar(frame, x, y0 + 48, bar_w, 8, beam_ratio, (255, 235, 110), "")
            status = " ULTRA" if self.ultra[player_id] else ""
            text = f"P{player_id + 1}{status} HP {self.hp[player_id]}  ENERGY {int(round(self.energy[player_id]))}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            tx = x if not align_right else x + bar_w - text_size[0]
            cv2.putText(frame, text, (tx, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 247, 250), 2, cv2.LINE_AA)

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

    @staticmethod
    def _draw_bar(
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        ratio: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        ratio = max(0.0, min(1.0, ratio))
        cv2.rectangle(frame, (x - 3, y - 3), (x + width + 3, y + height + 3), (10, 12, 16), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (54, 61, 72), -1)
        cv2.rectangle(frame, (x, y), (x + int(width * ratio), y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (230, 235, 245), 1)
        if label:
            cv2.putText(frame, label, (x + 6, y + height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 22, 28), 1, cv2.LINE_AA)

    def _beam_hits_chest(
        self,
        shooter: BeamState,
        target: BeamState,
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
