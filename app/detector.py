from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Optional

import cv2
import numpy as np


@dataclass
class BeamState:
    active: bool
    charging: bool
    confidence: float
    origin: tuple[int, int]
    direction: tuple[float, float]
    radius: int
    mode: str
    player_id: int = 0
    detected: bool = False
    chest_center: tuple[int, int] = (0, 0)
    chest_radius: int = 0
    charge_ratio: float = 0.0
    beam_ratio: float = 0.0
    powering: bool = False
    transforming: bool = False
    ultra: bool = False
    debug: dict[str, object] = field(default_factory=dict)


class SinglePoseDetector:
    def __init__(
        self,
        mp_pose,
        detection_confidence: float,
        tracking_confidence: float,
        player_id: int,
        visibility_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._mp_pose = mp_pose
        self._pose = self._mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._player_id = player_id
        self._charge_frames = 0
        self._active_smooth = 0.0
        self._last_direction = (1.0 if player_id == 0 else -1.0, 0.0)
        self._visibility_thresholds = {
            "wrist": 0.34,
            "elbow": 0.35,
            "shoulder": 0.38,
            "hip": 0.38,
        }
        if visibility_thresholds:
            self._visibility_thresholds.update(visibility_thresholds)

    def close(self) -> None:
        self._pose.close()

    def detect(self, frame_bgr: np.ndarray, x_offset: int = 0) -> BeamState:
        started = cv2.getTickCount()
        height, width = frame_bgr.shape[:2]
        roi = {"x": x_offset, "y": 0, "w": width, "h": height}
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        pose_ms = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
        if not result.pose_landmarks:
            self._charge_frames = max(0, self._charge_frames - 1)
            self._active_smooth *= 0.72
            return self._idle(
                width,
                height,
                x_offset,
                {
                    "roi": roi,
                    "pose_ms": round(pose_ms, 2),
                    "landmarks": False,
                    "missing": ["pose"],
                    "charge_frames": self._charge_frames,
                    "active_smooth": round(self._active_smooth, 3),
                },
            )

        landmarks = result.pose_landmarks.landmark
        pose = self._mp_pose.PoseLandmark

        tracked_landmarks = {
            "left_wrist": landmarks[pose.LEFT_WRIST],
            "right_wrist": landmarks[pose.RIGHT_WRIST],
            "left_elbow": landmarks[pose.LEFT_ELBOW],
            "right_elbow": landmarks[pose.RIGHT_ELBOW],
            "left_shoulder": landmarks[pose.LEFT_SHOULDER],
            "right_shoulder": landmarks[pose.RIGHT_SHOULDER],
            "left_hip": landmarks[pose.LEFT_HIP],
            "right_hip": landmarks[pose.RIGHT_HIP],
        }
        visibility = {name: round(float(landmark.visibility), 3) for name, landmark in tracked_landmarks.items()}

        left_wrist = self._point(tracked_landmarks["left_wrist"], width, height, self._visibility_threshold("left_wrist"))
        right_wrist = self._point(tracked_landmarks["right_wrist"], width, height, self._visibility_threshold("right_wrist"))
        left_elbow = self._point(tracked_landmarks["left_elbow"], width, height, self._visibility_threshold("left_elbow"))
        right_elbow = self._point(tracked_landmarks["right_elbow"], width, height, self._visibility_threshold("right_elbow"))
        left_shoulder = self._point(tracked_landmarks["left_shoulder"], width, height, self._visibility_threshold("left_shoulder"))
        right_shoulder = self._point(tracked_landmarks["right_shoulder"], width, height, self._visibility_threshold("right_shoulder"))
        left_hip = self._point(tracked_landmarks["left_hip"], width, height, self._visibility_threshold("left_hip"))
        right_hip = self._point(tracked_landmarks["right_hip"], width, height, self._visibility_threshold("right_hip"))

        important = [
            left_wrist,
            right_wrist,
            left_elbow,
            right_elbow,
            left_shoulder,
            right_shoulder,
        ]
        important_names = [
            "left_wrist",
            "right_wrist",
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
        ]
        missing = [name for name, point in zip(important_names, important) if point is None]
        if missing:
            self._active_smooth *= 0.75
            return self._idle(
                width,
                height,
                x_offset,
                {
                    "roi": roi,
                    "pose_ms": round(pose_ms, 2),
                    "landmarks": True,
                    "missing": missing,
                    "visibility": visibility,
                    "visibility_thresholds": self._visibility_thresholds,
                    "charge_frames": self._charge_frames,
                    "active_smooth": round(self._active_smooth, 3),
                },
            )

        lw = left_wrist
        rw = right_wrist
        le = left_elbow
        re = right_elbow
        ls = left_shoulder
        rs = right_shoulder
        assert lw and rw and le and re and ls and rs

        shoulder_mid = self._mid(ls, rs)
        hip_mid = self._mid(left_hip, right_hip) if left_hip and right_hip else shoulder_mid
        body_mid = self._mid(shoulder_mid, hip_mid)
        chest = self._mid(shoulder_mid, body_mid)
        hand_mid = self._mid(lw, rw)
        shoulder_width = max(64.0, self._dist(ls, rs))
        wrist_gap = self._dist(lw, rw)
        hands_together = wrist_gap < shoulder_width * 0.62

        reach = self._dist(body_mid, hand_mid)
        extension = reach / max(shoulder_width, 1.0)
        elbows_forward = self._dist(le, lw) + self._dist(re, rw) > shoulder_width * 0.95

        hands_low = lw[1] > shoulder_mid[1] + shoulder_width * 0.12 and rw[1] > shoulder_mid[1] + shoulder_width * 0.12
        wrists_at_chest_height = (
            abs(lw[1] - chest[1]) < shoulder_width * 0.45
            and abs(rw[1] - chest[1]) < shoulder_width * 0.45
        )
        wrists_near_core = self._dist(lw, body_mid) < shoulder_width * 1.12 and self._dist(rw, body_mid) < shoulder_width * 1.12
        elbows_tucked = self._dist(le, body_mid) < shoulder_width * 0.92 and self._dist(re, body_mid) < shoulder_width * 0.92
        powering = not hands_together and hands_low and wrists_at_chest_height and wrists_near_core and elbows_tucked
        transforming = lw[1] < shoulder_mid[1] - shoulder_width * 0.55 and rw[1] < shoulder_mid[1] - shoulder_width * 0.55

        charging = hands_together and extension < 1.25 and not powering and not transforming
        if charging:
            self._charge_frames = min(18, self._charge_frames + 1)
        else:
            self._charge_frames = max(0, self._charge_frames - 1)

        firing_gesture = hands_together and extension >= 0.72 and elbows_forward and not powering and not transforming
        active_raw = firing_gesture and (self._charge_frames >= 3 or extension >= 1.05)
        self._active_smooth = self._active_smooth * 0.68 + (1.0 if active_raw else 0.0) * 0.32
        active = self._active_smooth > 0.36 and not transforming

        dx = hand_mid[0] - shoulder_mid[0]
        dy = hand_mid[1] - shoulder_mid[1]
        norm = hypot(dx, dy)
        mode = "beam"
        if norm < shoulder_width * 0.32:
            mode = "front"
            direction = self._last_direction
        else:
            direction = (dx / norm, dy / norm)
            self._last_direction = direction

        confidence = min(1.0, 0.35 + self._active_smooth * 0.55 + min(extension / 2.0, 0.1))
        radius = int(max(24, min(96, shoulder_width * (0.28 + self._active_smooth * 0.22))))
        return BeamState(
            active=active,
            charging=charging or self._charge_frames > 0,
            confidence=confidence,
            origin=(int(hand_mid[0] + x_offset), int(hand_mid[1])),
            direction=direction,
            radius=radius,
            mode=mode,
            player_id=self._player_id,
            detected=True,
            chest_center=(int(chest[0] + x_offset), int(chest[1])),
            chest_radius=int(max(34, min(110, shoulder_width * 0.58))),
            powering=powering,
            transforming=transforming,
            debug={
                "roi": roi,
                "pose_ms": round(pose_ms, 2),
                "landmarks": True,
                "missing": [],
                "visibility": visibility,
                "visibility_thresholds": self._visibility_thresholds,
                "shoulder_width": round(shoulder_width, 1),
                "wrist_gap": round(wrist_gap, 1),
                "wrist_gap_ratio": round(wrist_gap / max(shoulder_width, 1.0), 3),
                "extension": round(extension, 3),
                "reach": round(reach, 1),
                "hands_together": hands_together,
                "elbows_forward": elbows_forward,
                "hands_low": hands_low,
                "wrists_at_chest_height": wrists_at_chest_height,
                "wrists_near_core": wrists_near_core,
                "elbows_tucked": elbows_tucked,
                "charging_raw": charging,
                "firing_extension_threshold": 0.72,
                "instant_fire_extension_threshold": 1.05,
                "firing_gesture": firing_gesture,
                "active_raw": active_raw,
                "charge_frames": self._charge_frames,
                "active_smooth": round(self._active_smooth, 3),
                "direction": (round(direction[0], 3), round(direction[1], 3)),
            },
        )

    def _idle(self, width: int, height: int, x_offset: int, debug: dict[str, object] | None = None) -> BeamState:
        return BeamState(
            active=False,
            charging=self._charge_frames > 0,
            confidence=0.0,
            origin=(x_offset + width // 2, height // 2),
            direction=self._last_direction,
            radius=32,
            mode="idle",
            player_id=self._player_id,
            detected=False,
            chest_center=(x_offset + width // 2, height // 2),
            chest_radius=0,
            debug=debug or {},
        )

    def _visibility_threshold(self, landmark_name: str) -> float:
        if "wrist" in landmark_name:
            return self._visibility_thresholds["wrist"]
        if "elbow" in landmark_name:
            return self._visibility_thresholds["elbow"]
        if "shoulder" in landmark_name:
            return self._visibility_thresholds["shoulder"]
        if "hip" in landmark_name:
            return self._visibility_thresholds["hip"]
        return self._visibility_thresholds["shoulder"]

    @staticmethod
    def _point(landmark, width: int, height: int, visibility_threshold: float) -> Optional[tuple[float, float]]:
        if landmark.visibility < visibility_threshold:
            return None
        return (landmark.x * width, landmark.y * height)

    @staticmethod
    def _mid(
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> tuple[float, float]:
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])


class BeamDetector:
    def __init__(
        self,
        detection_confidence: float = 0.55,
        tracking_confidence: float = 0.55,
        players: int = 2,
        wrist_visibility: float = 0.34,
        elbow_visibility: float = 0.35,
        shoulder_visibility: float = 0.38,
        hip_visibility: float = 0.38,
    ) -> None:
        import mediapipe as mp

        self.players = max(1, min(2, players))
        self._mp_pose = mp.solutions.pose
        visibility_thresholds = {
            "wrist": wrist_visibility,
            "elbow": elbow_visibility,
            "shoulder": shoulder_visibility,
            "hip": hip_visibility,
        }
        self._detectors = [
            SinglePoseDetector(
                self._mp_pose,
                detection_confidence,
                tracking_confidence,
                player_id=i,
                visibility_thresholds=visibility_thresholds,
            )
            for i in range(self.players)
        ]

    def close(self) -> None:
        for detector in self._detectors:
            detector.close()

    def detect(self, frame_bgr: np.ndarray) -> list[BeamState]:
        if self.players == 1:
            return [self._detectors[0].detect(frame_bgr, 0)]

        height, width = frame_bgr.shape[:2]
        overlap = int(width * 0.08)
        mid = width // 2
        left_end = min(width, mid + overlap)
        right_start = max(0, mid - overlap)

        left_state = self._detectors[0].detect(frame_bgr[:, :left_end], 0)
        right_state = self._detectors[1].detect(frame_bgr[:, right_start:], right_start)
        return [left_state, right_state]
