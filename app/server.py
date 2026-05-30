from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.beam import BeamController
from app.config import GameConfig
from app.detector import BeamDetector, BeamState
from app.effects import BeamEffect
from app.game import BeamGame
from app.npc import NpcConfig, NpcOpponent


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"


@dataclass
class StreamStats:
    frames_in: int = 0
    frames_out: int = 0
    last_frame_at: float = 0.0
    processing_ms: float = 0.0
    jpeg_quality: int = 82


class FrameProcessor:
    DEBUG_MODES = {"off", "basic", "full"}

    def __init__(self) -> None:
        self.game_config = GameConfig.from_env()
        requested_players = int(os.getenv("BEAM_PLAYERS", "2"))
        self.detector = BeamDetector(
            detection_confidence=float(os.getenv("BEAM_DETECTION", "0.55")),
            tracking_confidence=float(os.getenv("BEAM_TRACKING", "0.55")),
            players=requested_players,
            wrist_visibility=float(os.getenv("BEAM_WRIST_VISIBILITY", "0.34")),
            elbow_visibility=float(os.getenv("BEAM_ELBOW_VISIBILITY", "0.35")),
            shoulder_visibility=float(os.getenv("BEAM_SHOULDER_VISIBILITY", "0.38")),
            hip_visibility=float(os.getenv("BEAM_HIP_VISIBILITY", "0.38")),
        )
        self.npc_enabled = self.detector.players == 1 and os.getenv("BEAM_NPC", "1") != "0"
        self.npc = (
            NpcOpponent(
                NpcConfig(
                    cooldown_s=float(os.getenv("BEAM_NPC_COOLDOWN", "1.25")),
                    charge_s=float(os.getenv("BEAM_NPC_CHARGE", "1.25")),
                    attack_s=float(os.getenv("BEAM_NPC_ATTACK", "1.15")),
                )
            )
            if self.npc_enabled
            else None
        )
        if self.npc is not None:
            self.npc.set_difficulty(os.getenv("BEAM_NPC_DIFFICULTY", self.npc.difficulty))
        self.game_players = 2 if self.npc_enabled else self.detector.players
        self.beam = BeamController(self.game_players, self.game_config)
        self.effect = BeamEffect()
        self.game = BeamGame(
            players=self.game_players,
            config=self.game_config,
        )
        self.target_width = int(os.getenv("BEAM_WIDTH", "960"))
        self.stats = StreamStats(jpeg_quality=int(os.getenv("BEAM_JPEG_QUALITY", "82")))
        self._lock = threading.Lock()
        self._debug_mode = self._normalize_debug_mode(os.getenv("BEAM_DEBUG", "off"))
        self._battle_delay_s = float(os.getenv("BEAM_BATTLE_START_DELAY", "5"))
        self._battle_requested_at: float | None = None
        self._battle_started_at: float | None = None
        self._npc_ultra_applied = False
        self._latest_debug: list[dict[str, object]] = []
        self._latest_jpeg = self._encode_placeholder()

    def close(self) -> None:
        self.detector.close()

    def process_jpeg(self, payload: bytes) -> None:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return

        frame = self._resize(frame)
        raw_states = self.detector.detect(frame)
        battle_active, starts_in = self._update_battle_state()
        if self.npc is not None:
            npc_hp = self.game.hp[self.npc.config.player_id] if self.npc.config.player_id < len(self.game.hp) else 0
            raw_states.append(
                self.npc.state(
                    frame.shape,
                    raw_states[0] if raw_states else None,
                    npc_hp,
                    battle_active=battle_active,
                    starts_in=starts_in,
                )
            )
        states = self.beam.apply(raw_states, self.game.energy)
        collision = self.effect.beam_collision(frame.shape, states)
        self.game.update(states, frame.shape, collision)
        rendered = self.effect.render(frame, states, collision=collision)
        if self.npc is not None:
            npc_state = next((state for state in states if state.player_id == self.npc.config.player_id), None)
            if npc_state is not None:
                self.npc.draw(rendered, npc_state)
        rendered = self.game.draw_overlay(rendered, states)
        with self._lock:
            debug_mode = self._debug_mode
        if debug_mode != "off":
            self._draw_debug_overlay(rendered, states, debug_mode)
        ok, jpeg = cv2.imencode(
            ".jpg",
            rendered,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.stats.jpeg_quality],
        )
        if not ok:
            return

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._latest_jpeg = jpeg.tobytes()
            self.stats.frames_in += 1
            self.stats.frames_out += 1
            self.stats.last_frame_at = time.time()
            self.stats.processing_ms = self.stats.processing_ms * 0.85 + elapsed_ms * 0.15
            self._latest_debug = [self._debug_snapshot(state) for state in states]

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def snapshot_stats(self) -> dict[str, object]:
        with self._lock:
            age = time.time() - self.stats.last_frame_at if self.stats.last_frame_at else 9999.0
            return {
                "frames_in": self.stats.frames_in,
                "frames_out": self.stats.frames_out,
                "processing_ms": round(self.stats.processing_ms, 2),
                "camera_online": age < 2.5,
                "last_frame_age_s": round(age, 2),
                "target_width": self.target_width,
                "players": self.game_players,
                "detected_players": self.detector.players,
                "npc_enabled": self.npc_enabled,
                "npc": self._npc_snapshot_unlocked(),
                "battle": self._battle_snapshot_unlocked(age),
                "game": self.game.snapshot().__dict__,
                "debug_mode": self._debug_mode,
                "debug": self._latest_debug,
            }

    def set_debug_mode(self, mode: str) -> str:
        normalized = self._normalize_debug_mode(mode)
        with self._lock:
            self._debug_mode = normalized
        return normalized

    def set_npc_difficulty(self, difficulty: str) -> dict[str, object]:
        with self._lock:
            if self.npc is None:
                return {"enabled": False}
            selected = self.npc.set_difficulty(difficulty)
            self._battle_requested_at = None
            self._battle_started_at = None
            self._npc_ultra_applied = False
            player_id = self.npc.config.player_id
            if player_id < len(self.game.ultra):
                self.game.ultra[player_id] = False
            return {"enabled": True, "difficulty": selected, "starts_ultra": self.npc.starts_ultra}

    def start_battle(self, difficulty: str | None = None) -> dict[str, object]:
        with self._lock:
            if self.npc is not None and difficulty:
                self.npc.set_difficulty(difficulty)
            self.game.reset()
            self.beam.reset()
            if self.npc is not None:
                self.npc.reset()
            self._battle_requested_at = time.time() if self.npc_enabled else None
            self._battle_started_at = None
            self._npc_ultra_applied = False
            return self._battle_snapshot_unlocked(0.0)

    def reset_all(self) -> None:
        with self._lock:
            self.game.reset()
            self.beam.reset()
            if self.npc is not None:
                self.npc.reset()
            self._battle_requested_at = None
            self._battle_started_at = None
            self._npc_ultra_applied = False

    def snapshot_debug_settings(self) -> dict[str, object]:
        with self._lock:
            return {"mode": self._debug_mode, "modes": sorted(self.DEBUG_MODES)}

    def _update_battle_state(self) -> tuple[bool, float]:
        if not self.npc_enabled:
            return True, 0.0
        now = time.time()
        with self._lock:
            if self._battle_requested_at is None:
                return False, 0.0
            starts_in = self._battle_delay_s - (now - self._battle_requested_at)
            if starts_in > 0.0:
                return False, starts_in
            if self._battle_started_at is None:
                self._battle_started_at = now
                if self.npc is not None:
                    self.npc.reset()
                self._apply_npc_ultra_unlocked()
            return True, 0.0

    def _apply_npc_ultra_unlocked(self) -> None:
        if self._npc_ultra_applied or self.npc is None or not self.npc.starts_ultra:
            return
        player_id = self.npc.config.player_id
        if player_id < len(self.game.ultra):
            self.game.ultra[player_id] = True
        if player_id < len(self.game.energy):
            self.game.energy[player_id] = self.game.config.energy_max
        self._npc_ultra_applied = True

    def _npc_snapshot_unlocked(self) -> dict[str, object]:
        if self.npc is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "difficulty": self.npc.difficulty,
            "difficulties": ["easy", "normal", "hard"],
            "starts_ultra": self.npc.starts_ultra,
        }

    def _battle_snapshot_unlocked(self, _frame_age: float) -> dict[str, object]:
        if not self.npc_enabled:
            return {"enabled": False, "active": True, "starts_in": 0.0}
        now = time.time()
        requested = self._battle_requested_at is not None
        active = self._battle_started_at is not None
        starts_in = 0.0
        if requested and not active:
            starts_in = max(0.0, self._battle_delay_s - (now - (self._battle_requested_at or now)))
            if starts_in <= 0.0:
                self._battle_started_at = now
                if self.npc is not None:
                    self.npc.reset()
                self._apply_npc_ultra_unlocked()
                active = True
        return {
            "enabled": True,
            "requested": requested,
            "active": active,
            "starts_in": round(starts_in, 1),
            "delay_s": self._battle_delay_s,
        }

    @classmethod
    def _normalize_debug_mode(cls, mode: str) -> str:
        normalized = str(mode).lower().strip()
        return normalized if normalized in cls.DEBUG_MODES else "off"

    @staticmethod
    def _debug_snapshot(state: BeamState) -> dict[str, object]:
        return {
            "player_id": state.player_id,
            "detected": state.detected,
            "active": state.active,
            "charging": state.charging,
            "powering": state.powering,
            "transforming": state.transforming,
            "mode": state.mode,
            "confidence": round(state.confidence, 3),
            "origin": state.origin,
            "chest_center": state.chest_center,
            "debug": state.debug,
        }

    def _draw_debug_overlay(self, frame: np.ndarray, states: list[BeamState], mode: str) -> None:
        height, width = frame.shape[:2]
        colors = [(70, 210, 255), (245, 110, 230)]
        for state in states:
            color = colors[state.player_id % len(colors)]
            debug = state.debug
            roi = debug.get("roi") if isinstance(debug.get("roi"), dict) else None
            if roi:
                x = int(roi.get("x", 0))
                y = int(roi.get("y", 0))
                w = int(roi.get("w", width))
                h = int(roi.get("h", height))
                cv2.rectangle(frame, (x, y), (min(width - 1, x + w - 1), min(height - 1, y + h - 1)), color, 1)

            if state.detected:
                cv2.circle(frame, state.origin, 7, color, 2, cv2.LINE_AA)
                cv2.circle(frame, state.chest_center, max(4, state.chest_radius), color, 1, cv2.LINE_AA)

            panel_x = 12 if state.player_id == 0 else max(12, width - 332)
            panel_y = 70 + state.player_id * 158
            lines = self._debug_lines(state, mode)
            self._draw_text_panel(frame, panel_x, panel_y, lines, color)

    @staticmethod
    def _debug_lines(state: BeamState, mode: str) -> list[str]:
        debug = state.debug
        missing = debug.get("missing") or []
        lines = [
            f"P{state.player_id + 1} det={int(state.detected)} act={int(state.active)} chg={int(state.charging)}",
            f"pose={debug.get('pose_ms', '-')}ms mode={state.mode} conf={state.confidence:.2f}",
            f"missing={','.join(missing) if missing else '-'}",
            f"ext={debug.get('extension', '-')} gap={debug.get('wrist_gap_ratio', '-')} smooth={debug.get('active_smooth', '-')}",
            f"together={int(bool(debug.get('hands_together')))} chest={int(bool(debug.get('wrists_at_chest_height')))} fire={int(bool(debug.get('firing_gesture')))}",
        ]
        if mode == "full":
            visibility = debug.get("visibility") if isinstance(debug.get("visibility"), dict) else {}
            lines.extend([
                f"charge_frames={debug.get('charge_frames', '-')} shoulder={debug.get('shoulder_width', '-')}",
                f"power={int(state.powering)} ultra_pose={int(state.transforming)} elbows={int(bool(debug.get('elbows_forward')))}",
                "vis LW/RW/LE/RE",
                f"{visibility.get('left_wrist', '-')} {visibility.get('right_wrist', '-')} {visibility.get('left_elbow', '-')} {visibility.get('right_elbow', '-')}",
                "vis LS/RS/LH/RH",
                f"{visibility.get('left_shoulder', '-')} {visibility.get('right_shoulder', '-')} {visibility.get('left_hip', '-')} {visibility.get('right_hip', '-')}",
            ])
        return lines

    @staticmethod
    def _draw_text_panel(frame: np.ndarray, x: int, y: int, lines: list[str], color: tuple[int, int, int]) -> None:
        if not lines:
            return
        line_h = 18
        width = 310
        height = 12 + line_h * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (8, 12, 18), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x + 8, y + 20 + index * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 242, 250),
                1,
                cv2.LINE_AA,
            )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.target_width:
            return frame
        scale = self.target_width / float(width)
        return cv2.resize(frame, (self.target_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    def _encode_placeholder(self) -> bytes:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame[:] = (12, 14, 18)
        cv2.putText(
            frame,
            "Waiting for Windows camera stream...",
            (130, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Start SSH tunnel, then run windows_camera_client.py",
            (145, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 170, 190),
            1,
            cv2.LINE_AA,
        )
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return jpeg.tobytes() if ok else b""


processor: FrameProcessor | None = None
app = FastAPI(title="Beam Live")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    global processor
    processor = FrameProcessor()


@app.on_event("shutdown")
async def shutdown() -> None:
    if processor:
        processor.close()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/status")
async def status() -> dict[str, object]:
    assert processor is not None
    return processor.snapshot_stats()


@app.get("/debug")
async def get_debug() -> dict[str, object]:
    assert processor is not None
    return processor.snapshot_debug_settings()


@app.post("/debug")
async def set_debug(payload: dict[str, str]) -> dict[str, object]:
    assert processor is not None
    mode = processor.set_debug_mode(payload.get("mode", "off"))
    return {"ok": True, "mode": mode}


@app.post("/reset")
async def reset_game() -> dict[str, bool]:
    assert processor is not None
    processor.reset_all()
    return {"ok": True}


@app.post("/battle/start")
async def start_battle(payload: dict[str, str] | None = None) -> dict[str, object]:
    assert processor is not None
    battle = processor.start_battle((payload or {}).get("difficulty"))
    return {"ok": True, "battle": battle}


@app.post("/npc/difficulty")
async def set_npc_difficulty(payload: dict[str, str]) -> dict[str, object]:
    assert processor is not None
    npc = processor.set_npc_difficulty(payload.get("difficulty", "easy"))
    return {"ok": True, "npc": npc}


@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket) -> None:
    assert processor is not None
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_bytes()
            processor.process_jpeg(payload)
    except WebSocketDisconnect:
        return


@app.get("/video.mjpg")
async def video_mjpg() -> StreamingResponse:
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


async def _mjpeg_generator():
    assert processor is not None
    while True:
        jpeg = processor.latest_jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
        yield b"\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(1.0 / 30.0)
