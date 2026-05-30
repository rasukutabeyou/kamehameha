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
from app.detector import BeamDetector
from app.effects import BeamEffect
from app.game import BeamGame


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
    def __init__(self) -> None:
        self.game_config = GameConfig.from_env()
        self.detector = BeamDetector(
            detection_confidence=float(os.getenv("BEAM_DETECTION", "0.55")),
            tracking_confidence=float(os.getenv("BEAM_TRACKING", "0.55")),
            players=int(os.getenv("BEAM_PLAYERS", "2")),
        )
        self.beam = BeamController(self.detector.players, self.game_config)
        self.effect = BeamEffect()
        self.game = BeamGame(
            players=self.detector.players,
            config=self.game_config,
        )
        self.target_width = int(os.getenv("BEAM_WIDTH", "960"))
        self.stats = StreamStats(jpeg_quality=int(os.getenv("BEAM_JPEG_QUALITY", "82")))
        self._lock = threading.Lock()
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
        states = self.beam.apply(self.detector.detect(frame), self.game.energy)
        collision = self.effect.beam_collision(frame.shape, states)
        self.game.update(states, frame.shape, collision)
        rendered = self.effect.render(frame, states, collision=collision)
        rendered = self.game.draw_overlay(rendered, states)
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
                "players": self.detector.players,
                "game": self.game.snapshot().__dict__,
            }

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




@app.post("/reset")
async def reset_game() -> dict[str, bool]:
    assert processor is not None
    processor.game.reset()
    processor.beam.reset()
    return {"ok": True}


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
