import base64
import csv
import json
import math
import os
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from websocket import WebSocketApp

# ==============================
# Config
# ==============================
HTTP_BASE = "http://127.0.0.1:5000"
WS_URL = "ws://127.0.0.1:8080"

STEP_SIZE = 2.0
CAPTURE_TIMEOUT = 8.0
ARRIVAL_TIMEOUT = 4.0
MAX_STEPS = 350

# Navigation tuning
STEER_SMOOTH_ALPHA = 0.6
STUCK_WINDOW = 12
MIN_PROGRESS = 1.5
RECOVERY_BACK_DIST = 2.0
RECOVERY_TURN_DEG = 60.0

RUNS_CSV = os.path.join(os.path.dirname(__file__), "runs_log.csv")

# Four corners (explicit coordinates as requested)
CORNERS: Dict[str, Tuple[float, float]] = {
    "NE": (45.0, -45.0),
    "NW": (-45.0, -45.0),
    "SE": (45.0, 45.0),
    "SW": (-45.0, 45.0),
}


# ==============================
# Utilities
# ==============================
@dataclass
class Pose:
    x: float
    z: float


def data_url_to_bgr(data_url: str) -> Optional[np.ndarray]:
    try:
        _, b64 = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def heading_to_vector(theta_rad: float) -> Tuple[float, float]:
    return math.sin(theta_rad), math.cos(theta_rad)


def vector_to_heading(dx: float, dz: float) -> float:
    return math.atan2(dx, dz)


# ==============================
# API Client (Flask HTTP)
# ==============================
class ApiClient:
    def __init__(self, base: str = HTTP_BASE):
        self.base = base.rstrip("/")
        self.session = requests.Session()

    def post(self, path: str, json_data: dict) -> dict:
        r = self.session.post(f"{self.base}{path}", json=json_data, timeout=5)
        r.raise_for_status()
        return r.json()

    def get(self, path: str) -> dict:
        r = self.session.get(f"{self.base}{path}", timeout=5)
        r.raise_for_status()
        return r.json()

    def reset(self):
        return self.post("/reset", {})

    def set_goal_coords(self, x: float, z: float, y: float = 0.0):
        return self.post("/goal", {"x": float(x), "y": float(y), "z": float(z)})

    def move_abs(self, x: float, z: float):
        return self.post("/move", {"x": x, "z": z})

    def move_rel(self, turn_deg: float, distance: float):
        return self.post("/move_rel", {"turn": float(turn_deg), "distance": float(distance)})

    def stop(self):
        return self.post("/stop", {})

    def capture(self):
        return self.post("/capture", {})

    def collisions(self) -> int:
        return int(self.get("/collisions").get("count", 0))

    def ws_status(self) -> int:
        try:
            return int(self.get("/ws/status").get("clients", 0))
        except Exception:
            return 0


# ==============================
# WebSocket Client
# ==============================
class WSClient:
    def __init__(self, url: str = WS_URL):
        self.url = url
        self.ws: Optional[WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.connected = threading.Event()

        self.collision_count = 0
        self.goal_reached = threading.Event()
        self.arrived = threading.Event()

        self.pose_lock = threading.Lock()
        self.latest_pose: Optional[Pose] = None

        self.capture_queue: "queue.Queue[Tuple[np.ndarray, Pose]]" = queue.Queue(maxsize=5)

    def on_open(self, ws):
        self.connected.set()

    def on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except Exception:
            return

        if isinstance(data, dict) and data.get("type") == "collision" and data.get("collision"):
            self.collision_count += 1
            return

        if isinstance(data, dict) and data.get("type") == "goal_reached":
            self.goal_reached.set()
            return

        if isinstance(data, dict) and data.get("type") == "confirmation":
            if "Arrived at target" in data.get("message", ""):
                self.arrived.set()
            return

        if isinstance(data, dict) and data.get("type") == "capture_image_response":
            img_bgr = data_url_to_bgr(data.get("image", ""))
            pos = data.get("position", {}) or {}
            pose = Pose(float(pos.get("x", 0.0)), float(pos.get("z", 0.0)))
            with self.pose_lock:
                self.latest_pose = pose
            if img_bgr is not None:
                try:
                    self.capture_queue.put_nowait((img_bgr, pose))
                except queue.Full:
                    pass
            return

    def on_error(self, ws, error):
        pass

    def on_close(self, ws, code, msg):
        self.connected.clear()

    def start(self):
        self.ws = WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()
        self.connected.wait(timeout=5)

    def stop(self):
        try:
            if self.ws:
                self.ws.close()
        finally:
            self.connected.clear()


# ==============================
# Vision
# ==============================
class Vision:
    def __init__(self):
        self.lower = np.array([35, 60, 50])
        self.upper = np.array([90, 255, 255])

    def obstacle_mask(self, bgr: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def choose_steering(self, bgr: np.ndarray, desired_heading_rad: float) -> float:
        h, w = bgr.shape[:2]
        mask = self.obstacle_mask(bgr)
        roi_top = int(h * 0.4)
        roi = mask[roi_top:h, :]
        col_weights = roi.mean(axis=0)

        offsets_deg = np.linspace(-60, 60, 13)
        best_cost = float("inf")
        best_offset = 0.0

        def angle_to_col(offset_deg: float) -> int:
            frac = (offset_deg + 60.0) / 120.0
            return int(np.clip(frac * (w - 1), 0, w - 1))

        for off_deg in offsets_deg:
            c = angle_to_col(off_deg)
            band = max(20, w // 30)
            c0 = max(0, c - band)
            c1 = min(w - 1, c + band)
            obs_cost = float(col_weights[c0:c1 + 1].mean()) / 255.0
            dev_penalty = (abs(off_deg) / 60.0) ** 1.5
            cost = 1.6 * obs_cost + 0.4 * dev_penalty
            if cost < best_cost:
                best_cost = cost
                best_offset = math.radians(off_deg)

        return best_offset


# ==============================
# Navigator (single run)
# ==============================
class Navigator:
    def __init__(self, goal_name: str, goal_x: float, goal_z: float):
        self.api = ApiClient()
        self.ws = WSClient()
        self.vision = Vision()
        self.goal_name = goal_name
        self.goal = (float(goal_x), float(goal_z))
        self.prev_steer = 0.0
        self.dist_hist: deque[float] = deque(maxlen=STUCK_WINDOW)
        self.last_collision_count = 0
        self.steps_taken = 0

    def start(self) -> Tuple[bool, int, int]:
        print(f"[Navigator] Connecting WebSocket at {WS_URL} ...")
        self.ws.start()
        time.sleep(0.5)

        # Optional: wait for at least 1 simulator WS client
        for _ in range(10):
            if self.api.ws_status() >= 1:
                break
            time.sleep(0.5)

        # Do NOT reset here to avoid snapping back to origin between runs.
        # We'll compute collisions per-run as a delta instead of resetting the counter.
        # Clear any stale events/state from a previous run
        self.ws.goal_reached.clear()
        self.ws.arrived.clear()
        self.ws.collision_count = 0
        self.last_collision_count = 0
        self.dist_hist.clear()
        self.prev_steer = 0.0

        prev_collisions = 0
        try:
            prev_collisions = self.api.collisions()
        except Exception:
            prev_collisions = 0

        print(f"[Navigator] Setting goal {self.goal_name} at coords {self.goal}")
        try:
            self.api.set_goal_coords(self.goal[0], self.goal[1])
        except Exception as e:
            print(f"[ERROR] Failed to set goal: {e}")
            return False, 0, 0

        # Give simulator a moment to place goal and settle
        time.sleep(0.2)

        # Prime capture
        first_cap = self.request_capture_and_wait()
        if not first_cap:
            print("[INFO] Initial capture missing; will attempt to move cautiously.")

        self.navigate_loop()

        # Finalize
        try:
            self.api.stop()
        except Exception:
            pass

        # Per-run collisions as a delta from before the run
        collisions_http = 0
        try:
            now = self.api.collisions()
            collisions_http = max(0, now - prev_collisions)
        except Exception:
            pass
        reached = self.ws.goal_reached.is_set()
        return reached, collisions_http, self.steps_taken

    def request_capture_and_wait(self) -> Optional[Tuple[np.ndarray, Pose]]:
        try:
            resp = self.api.capture()
            if isinstance(resp, dict) and resp.get('error'):
                print(f"[WARN] /capture server error: {resp.get('error')}")
        except Exception as e:
            print(f"[WARN] /capture failed to send: {e}")
            return None
        try:
            img, pose = self.ws.capture_queue.get(timeout=CAPTURE_TIMEOUT)
            return img, pose
        except queue.Empty:
            print("[WARN] Timed out waiting for capture image")
            return None

    def navigate_loop(self):
        steps = 0
        while (not self.ws.goal_reached.is_set()) and steps < MAX_STEPS:
            steps += 1
            self.steps_taken = steps

            capture = self.request_capture_and_wait()
            if not capture:
                print("[INFO] No image. Sending small forward step.")
                self.safe_step_forward()
                continue

            img, pose = capture

            gx, gz = self.goal
            dx, dz = (gx - pose.x), (gz - pose.z)
            desired_theta = vector_to_heading(dx, dz)

            steer_raw = self.vision.choose_steering(img, desired_theta)
            steer_offset = (STEER_SMOOTH_ALPHA * self.prev_steer) + ((1 - STEER_SMOOTH_ALPHA) * steer_raw)
            self.prev_steer = steer_offset
            adjusted_theta = desired_theta + steer_offset

            step = STEP_SIZE
            vx, vz = heading_to_vector(adjusted_theta)
            target_x = pose.x + vx * step
            target_z = pose.z + vz * step

            try:
                self.ws.arrived.clear()
                self.api.move_abs(target_x, target_z)
            except Exception as e:
                print(f"[WARN] /move failed: {e}")
                time.sleep(0.2)
                continue

            t0 = time.time()
            while time.time() - t0 < ARRIVAL_TIMEOUT:
                if self.ws.goal_reached.is_set() or self.ws.arrived.is_set():
                    break
                time.sleep(0.05)

            goal_dist = math.hypot(gx - pose.x, gz - pose.z)
            self.dist_hist.append(goal_dist)

            if self.ws.goal_reached.is_set():
                break

            if self.ws.collision_count > self.last_collision_count:
                self.last_collision_count = self.ws.collision_count
                print("[RECOVER] Collision detected. Executing recovery maneuver.")
                self.recover_maneuver()
                continue

            if len(self.dist_hist) == STUCK_WINDOW and (self.dist_hist[0] - self.dist_hist[-1] < MIN_PROGRESS):
                print("[RECOVER] Low progress detected. Executing recovery maneuver.")
                self.recover_maneuver()
                self.prev_steer = 0.0
                self.dist_hist.clear()
        
        # Small dwell at goal to avoid immediate next-run motion looking odd
        if self.ws.goal_reached.is_set():
            time.sleep(1.0)

    def safe_step_forward(self):
        try:
            self.api.move_rel(0.0, 0.8)
        except Exception:
            pass
        time.sleep(0.2)

    def recover_maneuver(self):
        try:
            self.api.stop()
        except Exception:
            pass
        time.sleep(0.1)
        try:
            self.api.move_rel(0.0, -RECOVERY_BACK_DIST)
        except Exception:
            pass
        time.sleep(0.3)
        turn = RECOVERY_TURN_DEG if random.random() < 0.5 else -RECOVERY_TURN_DEG
        try:
            self.api.move_rel(turn, 1.5)
        except Exception:
            pass
        time.sleep(0.3)


# ==============================
# Runner for Level 1 (four corners)
# ==============================
def append_csv_row(path: str, row: List):
    header = ["timestamp", "goal_corner", "steps", "collisions", "goal_reached"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def average_collisions_from_csv(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    total = 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                total += int(row.get("collisions", 0))
                n += 1
            except Exception:
                pass
    return (total / n) if n > 0 else 0.0


def run_all_four():
    results = []
    for name, (x, z) in CORNERS.items():
        print("==================== New Run ====================")
        print(f"Corner: {name} @ ({x}, {z})")
        nav = Navigator(name, x, z)
        reached, collisions, steps = nav.start()
        print("==================== Result ====================")
        print(f"Corner: {name}")
        print(f"Steps taken: {steps}")
        print(f"Collisions (server count): {collisions}")
        print("SUCCESS: Goal reached!" if reached else "ENDED: Max steps or timeout.")
        results.append((name, steps, collisions, reached))
        # Log
        append_csv_row(RUNS_CSV, [int(time.time()), name, steps, collisions, int(reached)])
        # Short pause between runs to settle
        time.sleep(1.0)

    # Print average from CSV
    avg = average_collisions_from_csv(RUNS_CSV)
    print("==============================================")
    print(f"Average collisions across all logged runs: {avg:.2f}")


def main():
    run_all_four()


if __name__ == "__main__":
    main()
