#!/usr/bin/env python3
"""Play episode image data as side-by-side video.

Usage:
    python play-episode-video.py <episode_folder>
    python play-episode-video.py deploy_real/datasets/20260408_1437/episode_0000

Controls:
    SPACE   pause / resume
    q / ESC quit
    <- / -> step backward / forward (when paused)
    s       save video to episode folder
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_episode(episode_dir: Path):
    """Return sorted frame paths for each camera and metadata."""
    cameras = []
    for entry in sorted(episode_dir.iterdir()):
        if entry.is_dir():
            frames = sorted(entry.glob("*.jpg")) + sorted(entry.glob("*.png"))
            if frames:
                cameras.append((entry.name, frames))

    meta = {}
    json_path = episode_dir / "data.json"
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)

    return cameras, meta


def build_frame(cameras, idx: int, goal_text: str, show_info: bool) -> np.ndarray:
    """Build a single display frame by reading images from all cameras."""
    imgs = []
    for cam_name, frames in cameras:
        frame_idx = min(idx, len(frames) - 1)
        img = cv2.imread(str(frames[frame_idx]))
        if img is None:
            img = np.zeros((480, 480, 3), dtype=np.uint8)
        # Resize all camera feeds to the same height for side-by-side display
        imgs.append((cam_name, img))

    if not imgs:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Normalise heights
    target_h = max(im.shape[0] for _, im in imgs)
    resized = []
    for cam_name, im in imgs:
        h, w = im.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        im = cv2.resize(im, (new_w, target_h))
        if show_info:
            cv2.putText(im, cam_name, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        resized.append(im)

    combined = np.concatenate(resized, axis=1)

    if show_info and goal_text:
        max_text_w = combined.shape[1] - 16
        # Simple word-wrap by character width
        font_scale = 0.55
        thickness = 1
        line_h = 22
        words = goal_text.split()
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            tw, _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            if tw > max_text_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)

        overlay_h = line_h * len(lines) + 8
        overlay = combined[:overlay_h].copy()
        cv2.rectangle(overlay, (0, 0), (combined.shape[1], overlay_h), (0, 0, 0), -1)
        combined[:overlay_h] = cv2.addWeighted(combined[:overlay_h], 0.3, overlay, 0.7, 0)
        for i, line in enumerate(lines):
            cv2.putText(combined, line, (8, 18 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return combined


def play(episode_dir: Path, fps: float, save: bool, show_info: bool):
    cameras, meta = load_episode(episode_dir)

    if not cameras:
        print(f"No camera image folders found in {episode_dir}")
        sys.exit(1)

    n_frames = min(len(frames) for _, frames in cameras)
    fps = fps or meta.get("info", {}).get("image", {}).get("fps", 30)
    goal_text = meta.get("text", {}).get("goal", "")

    print(f"Episode : {episode_dir.name}")
    print(f"Cameras : {[c for c, _ in cameras]}")
    print(f"Frames  : {n_frames}")
    print(f"FPS     : {fps}")
    if goal_text:
        print(f"Goal    : {goal_text}")
    print("Controls: SPACE=pause  q/ESC=quit  ←/→=step  s=save")

    delay_ms = max(1, int(1000 / fps))
    paused = False
    idx = 0

    video_writer = None
    if save:
        first_frame = build_frame(cameras, 0, goal_text, show_info)
        h, w = first_frame.shape[:2]
        out_path = str(episode_dir / "episode_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Saving video to {out_path} ...")
        for i in range(n_frames):
            frame = build_frame(cameras, i, goal_text, show_info)
            video_writer.write(frame)
        video_writer.release()
        print("Video saved.")
        return

    window = f"Episode: {episode_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = build_frame(cameras, idx, goal_text, show_info)

        # Frame counter overlay
        if show_info:
            cv2.putText(frame, f"{idx + 1}/{n_frames}", (frame.shape[1] - 100, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            out_path = str(episode_dir / "episode_video.mp4")
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"Saving video to {out_path} ...")
            for i in range(n_frames):
                vw.write(build_frame(cameras, i, goal_text, show_info))
            vw.release()
            print("Video saved.")
        elif key == 81 or key == 2:  # left arrow
            idx = max(0, idx - 1)
            paused = True
        elif key == 83 or key == 3:  # right arrow
            idx = min(n_frames - 1, idx + 1)
            paused = True

        if not paused:
            idx += 1
            if idx >= n_frames:
                idx = 0  # loop

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play episode image data as video.")
    parser.add_argument("episode_dir", type=Path,
                        help="Path to episode folder, e.g. deploy_real/datasets/20260408_1437/episode_0000")
    parser.add_argument("--fps", type=float, default=0,
                        help="Playback FPS (default: read from data.json, fallback 30)")
    parser.add_argument("--save", action="store_true",
                        help="Save combined video as episode_video.mp4 instead of displaying")
    parser.add_argument("--no-info", action="store_true",
                        help="Hide overlaid text (camera names, goal, frame counter)")
    args = parser.parse_args()

    episode_dir = args.episode_dir
    if not episode_dir.is_absolute():
        episode_dir = Path(os.getcwd()) / episode_dir

    if not episode_dir.exists():
        print(f"Episode folder not found: {episode_dir}")
        sys.exit(1)

    play(episode_dir, args.fps, args.save, not args.no_info)


if __name__ == "__main__":
    main()
