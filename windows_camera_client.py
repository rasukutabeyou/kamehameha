from __future__ import annotations

import argparse
import asyncio
import time

import cv2
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Windows webcam stream to Kamehameha Live.")
    parser.add_argument("--server", default="ws://127.0.0.1:8000/ws/camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--quality", type=int, default=78)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


async def stream_camera(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    interval = 1.0 / max(args.fps, 1.0)

    try:
        while True:
            try:
                async with websockets.connect(args.server, max_size=None, ping_interval=20) as ws:
                    print(f"connected: {args.server}")
                    while True:
                        started = time.perf_counter()
                        ok, frame = cap.read()
                        if not ok:
                            await asyncio.sleep(0.1)
                            continue
                        if args.mirror:
                            frame = cv2.flip(frame, 1)
                        ok, jpeg = cv2.imencode(
                            ".jpg",
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
                        )
                        if ok:
                            await ws.send(jpeg.tobytes())
                        elapsed = time.perf_counter() - started
                        await asyncio.sleep(max(0.0, interval - elapsed))
            except (OSError, websockets.WebSocketException) as exc:
                print(f"disconnected: {exc}; retrying in 2s")
                await asyncio.sleep(2.0)
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    asyncio.run(stream_camera(args))


if __name__ == "__main__":
    main()
