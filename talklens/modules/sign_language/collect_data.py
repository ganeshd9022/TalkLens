"""
Data Collection Script
───────────────────────
Run this script to record hand landmark sequences for each ASL sign.
Usage:
    python collect_data.py --signs A B C D --samples 200 --output data/raw
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


def parse_args():
    p = argparse.ArgumentParser(description="Collect ASL landmark sequences.")
    p.add_argument("--signs",   nargs="+", required=True, help="Signs to record e.g. A B C")
    p.add_argument("--samples", type=int,  default=200,   help="Samples per sign")
    p.add_argument("--seq_len", type=int,  default=30,    help="Frames per sequence")
    p.add_argument("--output",  type=str,  default="data/raw")
    return p.parse_args()


def main():
    args  = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    hands     = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    label_map = {}

    for idx, sign in enumerate(args.signs):
        label_map[idx] = sign
        sign_dir = outdir / sign
        sign_dir.mkdir(exist_ok=True)

        collected = 0
        print(f"\n[COLLECTION] Sign: '{sign}' | Hold and press SPACE to start.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(
                display,
                f"Ready for '{sign}'. Press SPACE.", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )
            cv2.imshow("TalkLens — Data Collection", display)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        print(f"  Recording {args.samples} samples for '{sign}'…")
        sequence: list = []

        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            landmarks = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()

            if landmarks is not None:
                sequence.append(landmarks)

            if len(sequence) == args.seq_len:
                fname = sign_dir / f"{collected:04d}.npy"
                np.save(str(fname), np.array(sequence))
                sequence = []
                collected += 1
                print(f"    Sample {collected}/{args.samples}", end="\r")

            status = f"'{sign}': {collected}/{args.samples}"
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("TalkLens — Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(f"\n  ✓ Done collecting '{sign}'.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    with open(outdir.parent / "processed" / "labels.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[DONE] Labels saved. Raw data at: {outdir}")


if __name__ == "__main__":
    main()
