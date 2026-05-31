from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.detector import BeamState


@dataclass
class Explosion:
    center: tuple[int, int]
    strength: float
    ttl: int = 14
    age: int = 0


class BeamEffect:
    def __init__(self) -> None:
        self._frame_index = 0
        self._explosions: list[Explosion] = []

    def render(
        self,
        frame: np.ndarray,
        states: BeamState | list[BeamState],
        collision: tuple[tuple[int, int], float] | None = None,
    ) -> np.ndarray:
        self._frame_index += 1
        output = frame.copy()
        state_list = states if isinstance(states, list) else [states]

        for state in state_list:
            if state.ultra:
                output = self._draw_ultra(output, state)
            if state.powering:
                output = self._draw_power(output, state)
            if state.charging:
                output = self._draw_charge(output, state)

        if collision is None:
            collision = self.beam_collision(output.shape, state_list)
        if collision is not None:
            center, strength = collision
            self._explosions.append(Explosion(center=center, strength=strength, ttl=14))

        for state in state_list:
            if state.active:
                if state.mode == "front":
                    output = self._draw_front_burst(output, state)
                else:
                    output = self._draw_beam(output, state, collision[0] if collision else None)
                output = self._draw_particles(output, state)

        output = self._draw_explosions(output)
        return output

    def _draw_ultra(self, frame: np.ndarray, state: BeamState) -> np.ndarray:
        center = state.chest_center
        span = max(42, state.chest_radius)
        palette = [(20, 210, 255), (70, 245, 255), (210, 255, 255)]
        for i in range(14):
            angle = i * math.tau / 14.0 + self._frame_index * 0.05
            wave = 0.85 + 0.18 * math.sin(self._frame_index * 0.18 + i)
            inner = span * 0.65
            outer = span * (1.45 + (i % 4) * 0.16) * wave
            p1 = (int(center[0] + math.cos(angle) * inner), int(center[1] + math.sin(angle) * inner))
            p2 = (int(center[0] + math.cos(angle) * outer), int(center[1] + math.sin(angle) * outer))
            cv2.line(frame, p1, p2, palette[i % len(palette)], 2, lineType=cv2.LINE_AA)
        return frame


    def _draw_power(self, frame: np.ndarray, state: BeamState) -> np.ndarray:
        center = state.chest_center
        palette = [(45, 210, 100), (105, 255, 155), (225, 255, 210)]
        span = max(36, state.chest_radius)
        for i in range(8):
            offset = (i - 3.5) / 3.5
            wave = math.sin(self._frame_index * 0.22 + i * 0.9 + state.player_id)
            x = int(center[0] + offset * span * 0.9 + wave * 5)
            y0 = int(center[1] + span * 0.78)
            y1 = int(center[1] - span * (0.85 + (i % 3) * 0.16))
            cv2.line(frame, (x, y0), (x, y1), palette[i % len(palette)], 2, lineType=cv2.LINE_AA)
        return frame

    def _draw_charge(self, frame: np.ndarray, state: BeamState) -> np.ndarray:
        overlay = np.zeros_like(frame)
        pulse = 0.75 + 0.25 * math.sin(time.time() * 18.0 + state.player_id * 1.7)
        charge_boost = 0.45 + state.charge_ratio * 0.55
        radius = int(state.radius * (0.8 + state.confidence * pulse * charge_boost))
        center = state.origin

        palette = self._palette(state.player_id)
        for scale, color, alpha in [
            (1.7, palette[0], 0.08 + state.charge_ratio * 0.05),
            (1.2, palette[1], 0.12 + state.charge_ratio * 0.10),
            (0.72, (255, 255, 245), 0.22 + state.charge_ratio * 0.28),
        ]:
            cv2.circle(overlay, center, int(radius * scale), color, -1, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)
            overlay[:] = 0

        for i in range(10):
            angle = i * math.tau / 16.0 + self._frame_index * 0.13 * (-1 if state.player_id else 1)
            outer = (
                int(center[0] + math.cos(angle) * radius * 1.7),
                int(center[1] + math.sin(angle) * radius * 1.7),
            )
            cv2.line(frame, outer, center, palette[1], 2, lineType=cv2.LINE_AA)
        self._draw_radial_meter(frame, center, int(radius * 1.95), state.charge_ratio, palette[2])
        return frame

    def _draw_beam(
        self,
        frame: np.ndarray,
        state: BeamState,
        collision_center: tuple[int, int] | None = None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        ox, oy = state.origin
        dx, dy = state.direction
        length = int(math.hypot(width, height) * 1.35)
        end = (int(ox + dx * length), int(oy + dy * length))
        if collision_center is not None:
            end = collision_center
        perp = (-dy, dx)
        base_width = int(state.radius * (0.82 + state.confidence * 0.72))
        palette = self._palette(state.player_id)

        glow = np.zeros_like(frame)
        for spread, color, alpha in [
            (2.2, palette[0], 0.10),
            (1.45, palette[1], 0.20),
            (0.78, palette[2], 0.38),
            (0.34, (255, 255, 255), 0.62),
        ]:
            half = base_width * spread
            wobble = math.sin(self._frame_index * 0.45 + state.player_id * 2.4) * base_width * 0.18
            p1 = (int(ox + perp[0] * (half + wobble)), int(oy + perp[1] * (half + wobble)))
            p2 = (int(ox - perp[0] * (half - wobble)), int(oy - perp[1] * (half - wobble)))
            p3 = (int(end[0] - perp[0] * half * 0.42), int(end[1] - perp[1] * half * 0.42))
            p4 = (int(end[0] + perp[0] * half * 0.42), int(end[1] + perp[1] * half * 0.42))
            cv2.fillConvexPoly(glow, np.array([p1, p4, p3, p2], dtype=np.int32), color, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1.0, glow, alpha, 0.0)
            glow[:] = 0

        cv2.circle(frame, state.origin, int(base_width * 0.78), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, state.origin, int(base_width * 1.05), palette[1], 3, lineType=cv2.LINE_AA)
        return frame

    def _draw_front_burst(self, frame: np.ndarray, state: BeamState) -> np.ndarray:
        center = state.origin
        overlay = np.zeros_like(frame)
        pulse = 1.0 + 0.12 * math.sin(self._frame_index * 0.7)
        palette = self._palette(state.player_id)
        for scale, color, alpha in [
            (3.2, palette[0], 0.14),
            (2.1, palette[1], 0.24),
            (1.05, palette[2], 0.48),
            (0.52, (255, 255, 255), 0.70),
        ]:
            cv2.circle(overlay, center, int(state.radius * scale * pulse), color, -1, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)
            overlay[:] = 0
        for i in range(22):
            angle = i * math.tau / 36.0 + self._frame_index * 0.09
            inner = state.radius * 1.3
            outer = state.radius * (3.9 + (i % 5) * 0.25)
            p1 = (int(center[0] + math.cos(angle) * inner), int(center[1] + math.sin(angle) * inner))
            p2 = (int(center[0] + math.cos(angle) * outer), int(center[1] + math.sin(angle) * outer))
            cv2.line(frame, p1, p2, palette[1], 3, lineType=cv2.LINE_AA)
        return frame

    def _draw_particles(self, frame: np.ndarray, state: BeamState) -> np.ndarray:
        rng = np.random.default_rng(self._frame_index * 17 + state.player_id)
        ox, oy = state.origin
        dx, dy = state.direction
        perp = np.array([-dy, dx])
        direction = np.array([dx, dy])
        count = 22
        palette = self._palette(state.player_id)
        for _ in range(count):
            distance = rng.uniform(-state.radius * 0.35, state.radius * 5.4)
            side = rng.normal(0, state.radius * 0.72)
            point = np.array([ox, oy], dtype=np.float32) + direction * distance + perp * side
            size = int(rng.integers(1, 5))
            color = palette[int(rng.integers(0, len(palette)))]
            cv2.circle(frame, (int(point[0]), int(point[1])), size, color, -1, lineType=cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_radial_meter(
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        ratio: float,
        color: tuple[int, int, int],
    ) -> None:
        ratio = max(0.0, min(1.0, ratio))
        cv2.circle(frame, center, radius, (40, 46, 58), 2, lineType=cv2.LINE_AA)
        if ratio <= 0.0:
            return
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            360 * ratio,
            color,
            4,
            lineType=cv2.LINE_AA,
        )

    def _draw_explosions(self, frame: np.ndarray) -> np.ndarray:
        next_explosions: list[Explosion] = []
        for explosion in self._explosions[-4:]:
            progress = explosion.age / max(1, explosion.ttl)
            fade = max(0.0, 1.0 - progress)
            radius = int((30 + explosion.strength * 86) * (0.32 + progress * 1.45))
            center = explosion.center
            overlay = np.zeros_like(frame)

            for scale, color, alpha in [
                (1.55, (255, 70, 20), 0.17 * fade),
                (1.0, (255, 215, 40), 0.31 * fade),
                (0.45, (255, 255, 255), 0.58 * fade),
            ]:
                cv2.circle(overlay, center, int(radius * scale), color, -1, lineType=cv2.LINE_AA)
                frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)
                overlay[:] = 0

            spikes = 16
            for i in range(spikes):
                angle = i * math.tau / spikes + self._frame_index * 0.08
                inner = radius * 0.25
                outer = radius * (1.1 + (i % 4) * 0.22)
                p1 = (int(center[0] + math.cos(angle) * inner), int(center[1] + math.sin(angle) * inner))
                p2 = (int(center[0] + math.cos(angle) * outer), int(center[1] + math.sin(angle) * outer))
                cv2.line(frame, p1, p2, (255, 245, 120), max(1, int(3 * fade)), lineType=cv2.LINE_AA)

            explosion.age += 1
            if explosion.age < explosion.ttl:
                next_explosions.append(explosion)
        self._explosions = next_explosions
        return frame

    def beam_collision(
        self,
        shape: tuple[int, int, int],
        states: list[BeamState],
    ) -> tuple[tuple[int, int], float] | None:
        active = [state for state in states if state.active and state.mode == "beam"]
        if len(active) < 2:
            return None

        height, width = shape[:2]
        length = math.hypot(width, height) * 1.35
        best: tuple[tuple[int, int], float] | None = None
        for i, first in enumerate(active):
            for second in active[i + 1 :]:
                if self._dot(first.direction, second.direction) > -0.28:
                    continue
                p = np.array(first.origin, dtype=np.float32)
                r = np.array(first.direction, dtype=np.float32) * length
                q = np.array(second.origin, dtype=np.float32)
                s = np.array(second.direction, dtype=np.float32) * length
                hit = self._segment_intersection(p, p + r, q, q + s)
                if hit is None:
                    hit = self._closest_midpoint(p, p + r, q, q + s)
                center, distance = hit
                threshold = first.radius * 2.8 + second.radius * 2.8
                if distance <= threshold:
                    x, y = int(center[0]), int(center[1])
                    if -50 <= x <= width + 50 and -50 <= y <= height + 50:
                        strength = max(0.35, 1.0 - distance / max(1.0, threshold))
                        best = ((x, y), strength)
        return best

    @staticmethod
    def _segment_intersection(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(float(denom)) < 1e-5:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        point = np.array([px, py], dtype=np.float32)
        if _inside(point, p1, p2) and _inside(point, p3, p4):
            return point, 0.0
        return None

    @staticmethod
    def _closest_midpoint(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        candidates = []
        for point, a, b in [(p1, p3, p4), (p2, p3, p4), (p3, p1, p2), (p4, p1, p2)]:
            projected = _project_to_segment(point, a, b)
            distance = float(np.linalg.norm(point - projected))
            candidates.append(((point + projected) * 0.5, distance))
        return min(candidates, key=lambda item: item[1])

    @staticmethod
    def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def _palette(player_id: int) -> list[tuple[int, int, int]]:
        if player_id == 1:
            return [(255, 40, 145), (255, 95, 220), (255, 205, 255)]
        return [(255, 85, 10), (255, 220, 40), (255, 255, 160)]


def _project_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-5:
        return start
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    return start + segment * t


def _inside(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    margin = 1.0
    return bool(
        min(start[0], end[0]) - margin <= point[0] <= max(start[0], end[0]) + margin
        and min(start[1], end[1]) - margin <= point[1] <= max(start[1], end[1]) + margin
    )
