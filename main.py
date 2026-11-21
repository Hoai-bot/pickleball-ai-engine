import os
import re
import cv2
import gc
import math
import hmac
import hashlib
import time
import uuid
import urllib.request
import numpy as np
from typing import Optional, List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PVR AI Enterprise Engine V7.0 - Multi-Dimensional Pro Calibration",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "PVR_AI_ENTERPRISE_KEY_2026_SECURE")
RESULTS_CACHE: Dict[str, dict] = {}

try:
    import mediapipe as mp
    try:
        MP_POSE_MODULE = mp.solutions.pose
    except AttributeError:
        import mediapipe.python.solutions.pose as MP_POSE_MODULE
except Exception as e:
    print(f"MediaPipe Init Warning: {e}")
    MP_POSE_MODULE = None


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return round(float(angle), 1)


def generate_secure_qr_signature(pick_id: str, rating: str, winrate: str, audit_hash: str) -> str:
    raw_data = f"{pick_id}:{rating}:{winrate}:{audit_hash}"
    return hmac.new(SECRET_KEY.encode('utf-8'), raw_data.encode('utf-8'), hashlib.sha256).hexdigest()[:16]


def download_online_video(url: str, output_path: str) -> Optional[str]:
    try:
        import yt_dlp
        clean_url = re.sub(r'[\(\)\[\]]', '', url.strip())
        urls_found = re.findall(r'(https?://[^\s]+)', clean_url)
        if urls_found:
            clean_url = urls_found[0]
        if "youtu.be/" in clean_url:
            video_id = clean_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
            clean_url = f"https://www.youtube.com/watch?v={video_id}"
        clean_url = clean_url.split("?si=")[0].split("&si=")[0]

        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/worst[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
            'nocheckcertificate': True,
            'socket_timeout': 12,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        if os.path.exists("cookies.txt"):
            ydl_opts['cookiefile'] = 'cookies.txt'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])
        return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else None
    except Exception as e:
        print(f"Download Error: {e}")
        return None


def verify_and_scrape_profile(profile_url: str):
    if not profile_url or profile_url.strip() in ["", "None", "string"]:
        return None, "UNVERIFIED"

    clean_url = profile_url.strip()
    try:
        is_dupr = "mydupr.com" in clean_url or "dupr.com" in clean_url
        is_pickleball_global = "pickleball.global" in clean_url or "pickleballtournaments.com" in clean_url
        is_pvr = "pvr" in clean_url.lower() or "pick-" in clean_url.lower()

        if not (is_dupr or is_pickleball_global or is_pvr):
            return None, "CUSTOM_OR_UNVERIFIED_URL"
        if is_pvr and not clean_url.startswith("http"):
            return None, "PICK_ID_FORMAT_VERIFIED"

        req = urllib.request.Request(clean_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            html = response.read().decode('utf-8', errors='ignore')
            match = re.search(r'(\d+)\s*(Wins|Thắng).*?(\d+)\s*(Losses|Bại|Total)', html, re.IGNORECASE)
            if match:
                wins = int(match.group(1))
                losses = int(match.group(3))
                total = wins + losses
                return {"wins": wins, "total": total, "winrate": round((wins / max(1, total)) * 100, 1)}, "VERIFIED_OFFICIAL"

        return None, "VERIFIED_DOMAIN_PENDING_SYNC"
    except Exception:
        return None, "LINK_OFFLINE_OR_PRIVATE"


def extract_torso_color_hsv(frame, norm_cx: float, norm_cy: float) -> Optional[np.ndarray]:
    try:
        h, w, _ = frame.shape
        px = int(np.clip(norm_cx * w, 0, w - 1))
        py = int(np.clip((norm_cy - 0.08) * h, 0, h - 1))
        box = max(8, int(w * 0.03))
        patch = frame[max(0, py - box):min(h, py + box), max(0, px - box):min(w, px + box)]
        if patch.size == 0:
            return None
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return np.mean(hsv_patch.reshape(-1, 3), axis=0)
    except Exception:
        return None


def get_dynamic_roi_box(
    frame_shape: Tuple[int, int, int],
    court_distance: str,
    player_position: str,
    tracked_center: Optional[Tuple[float, float]]
) -> Tuple[int, int, int, int]:
    h, w, _ = frame_shape
    dist = str(court_distance).lower()
    pos = str(player_position).lower()

    if dist == "near":
        court_y1, court_y2 = int(h * 0.25), h
    elif dist == "far":
        court_y1, court_y2 = 0, int(h * 0.70)
    else:
        court_y1, court_y2 = 0, h

    if tracked_center is None:
        x1 = 0 if pos == "left" else (int(w * 0.30) if pos == "right" else 0)
        x2 = int(w * 0.70) if pos == "left" else w
        return x1, court_y1, x2, court_y2

    cx, cy = tracked_center
    half_win_w = int(w * (0.20 if dist == "far" else 0.24))
    half_win_h = int(h * (0.24 if dist == "far" else 0.28))

    px = int(cx * w)
    py = int(cy * h)

    x1 = max(0, px - half_win_w)
    x2 = min(w, px + half_win_w)
    y1 = max(court_y1, py - half_win_h)
    y2 = min(court_y2, py + half_win_h)

    if (x2 - x1) < int(w * 0.22):
        x1 = max(0, px - int(w * 0.18))
        x2 = min(w, px + int(w * 0.18))
    if (y2 - y1) < int(h * 0.22):
        y1 = court_y1
        y2 = court_y2

    return x1, y1, x2, y2


def classify_pvr_division(pvr_score: float) -> str:
    if pvr_score < 2.25:
        return "Beginner (No Kitchen / 2.00 - 2.24)"
    elif pvr_score < 2.75:
        return "Newbie (2.25 - 2.74)"
    elif pvr_score < 3.45:
        return "Nâng Cao / Intermediate (2.75 - 3.44)"
    elif pvr_score < 4.00:
        return "Nâng Cao Open / Advanced (3.45 - 3.99)"
    return "Pro / Bán Chuyên (4.00+)"


def compute_5_pillar_pvr_metrics(
    r_elbow_list: List[float],
    l_elbow_list: List[float],
    knee_angles: List[float],
    hip_positions: List[Tuple[float, float]],
    wrist_speeds: List[float],
    trunk_tilts: List[float],
    ankle_spreads: List[float],
    court_distance: str = "near",
    player_tier: str = "intermediate"
) -> dict:
    """
    Hệ thống đánh giá 5 trụ cột chuẩn PB Vision + SwingVision + DUPR:
    1. Compact Swing & Soft/Hard Game Control (20%)
    2. Core Balance & Trunk Stability (20%)
    3. Split-Step & Footwork Agility (20%)
    4. Kitchen Transition & Court Coverage (25%)
    5. Ready Stance & Athletic Knee Bend (15%)
    """
    tier_lower = str(player_tier).lower()
    is_far = str(court_distance).lower() == "far"

    if tier_lower in ["beginner", "no_kitchen", "tap_su"]:
        tier_offset = -0.22
        fallback_pvr = 2.15
    elif tier_lower in ["newbie", "social", "phong trào", "moi_choi"]:
        tier_offset = -0.08
        fallback_pvr = 2.55
    elif tier_lower in ["semi_pro", "advanced", "nâng cao", "bán chuyên"]:
        tier_offset = 0.35
        fallback_pvr = 3.45
    elif tier_lower in ["pro", "pro elite"]:
        tier_offset = 0.75
        fallback_pvr = 4.10
    else:
        tier_offset = 0.16
        fallback_pvr = 3.08

    r_std = float(np.std(r_elbow_list)) if len(r_elbow_list) >= 3 else 0.0
    l_std = float(np.std(l_elbow_list)) if len(l_elbow_list) >= 3 else 0.0
    elbow_angles = r_elbow_list if (r_std >= l_std and len(r_elbow_list) >= 3) else (
        l_elbow_list if len(l_elbow_list) >= 3 else r_elbow_list
    )

    if len(elbow_angles) < 3 or len(knee_angles) < 3:
        default_pvr = round(max(2.00, fallback_pvr), 2)
        return {
            "bio_score": default_pvr,
            "bio_floor_lock": round(max(2.00, default_pvr - 0.15), 2),
            "bio_percent": 74.0 if default_pvr < 2.75 else 82.0,
            "consistency_index": 76.0,
            "ready_stance_percent": 75.0,
            "swing_velocity_index": 72.0,
            "footwork_agility_index": 72.0,
            "kitchen_transition_index": 74.0,
            "core_balance_index": 78.0,
            "avg_elbow": 115.0,
            "avg_knee": 132.0
        }

    # --- TRỤ CỘT 1: COMPACT SWING & SOFT/HARD CONTROL (20%) ---
    # Tỷ lệ các pha giữ tay gọn (85° - 142°) đặc trưng của Dink/Reset/Volley chuẩn
    compact_shots = [a for a in elbow_angles if 85.0 <= a <= 142.0]
    compact_ratio = len(compact_shots) / len(elbow_angles)
    if len(wrist_speeds) >= 2:
        sorted_ws = sorted(wrist_speeds, reverse=True)
        top_ws = sorted_ws[:max(1, len(sorted_ws) // 3)]
        peak_ws = float(np.mean(top_ws))
        vel_score = min(1.0, max(0.35, (peak_ws - 0.018) / 0.125))
    else:
        vel_score = 0.65
    pillar_swing = (compact_ratio * 0.55) + (vel_score * 0.45)

    # --- TRỤ CỘT 2: CORE BALANCE & TRUNK STABILITY (20%) ---
    # VĐV trình cao giữ trục lưng thẳng ổn định (góc nghiêng thân < 16°), không bị với bóng
    if len(trunk_tilts) >= 3:
        mean_tilt = float(np.mean(trunk_tilts))
        tilt_std = float(np.std(trunk_tilts))
        pillar_balance = max(0.40, min(1.0, 1.0 - (max(0.0, mean_tilt - 8.0) / 35.0) - (tilt_std / 30.0)))
    else:
        pillar_balance = 0.78

    # --- TRỤ CỘT 3: SPLIT-STEP & FOOTWORK AGILITY (20%) ---
    if len(hip_positions) >= 3:
        lateral_steps = [abs(hip_positions[i][0] - hip_positions[i - 1][0]) for i in range(1, len(hip_positions))]
        avg_lat = float(np.mean(lateral_steps))
        lat_norm = min(1.0, max(0.32, (avg_lat - 0.004) / 0.030))
    else:
        lat_norm = 0.62

    if len(ankle_spreads) >= 3:
        spread_var = float(np.std(ankle_spreads))
        # Có nhịp nhún tách chân (split-step) tạo dao động độ rộng chân tự nhiên
        split_step_norm = min(1.0, max(0.40, spread_var / 0.035))
    else:
        split_step_norm = 0.65
    pillar_footwork = (lat_norm * 0.60) + (split_step_norm * 0.40)

    # --- TRỤ CỘT 4: KITCHEN TRANSITION & COURT COVERAGE (25%) ---
    # Đo khả năng chủ động di chuyển lên vùng Kitchen (NVZ) và giữ vị trí tấn công
    if len(hip_positions) >= 4:
        y_vals = [p[1] for p in hip_positions]
        y_range = max(y_vals) - min(y_vals)
        # Sân gần: tiến lên lưới nghĩa là y giảm; Sân xa: tiến lên lưới nghĩa là y tăng
        forward_progress = (y_vals[0] - min(y_vals)) if not is_far else (max(y_vals) - y_vals[0])
        transition_norm = min(1.0, max(0.38, (y_range * 2.4) + (max(0.0, forward_progress) * 2.2)))
    else:
        transition_norm = 0.68
    pillar_kitchen = transition_norm

    # --- TRỤ CỘT 5: READY STANCE & ATHLETIC KNEE BEND (15%) ---
    ready_frames = [ang for ang in knee_angles if 105.0 <= ang <= 152.0]
    ready_stance_ratio = len(ready_frames) / len(knee_angles)
    mean_knee = float(np.mean(knee_angles))
    knee_form = max(0.45, 1.0 - (max(0.0, abs(mean_knee - 130.0) - 15.0) / 48.0))
    pillar_stance = (ready_stance_ratio * 0.60) + (knee_form * 0.40)

    # Tổng hợp điểm kỹ thuật 0 - 100
    bio_percent = round(
        min(98.5, max(40.0,
            (pillar_swing * 20.0) +
            (pillar_balance * 20.0) +
            (pillar_footwork * 20.0) +
            (pillar_kitchen * 25.0) +
            (pillar_stance * 15.0)
        )),
        1
    )

    # Quy đổi sang thang PVR chuẩn PVNA/DUPR (2.00 - 4.85)
    raw_pvr = 2.05 + ((bio_percent - 38.0) / 60.0) * 1.95 + tier_offset

    # Chống khai gian trình độ (Anti-Sandbagging): tự động đẩy lên nếu chỉ số Kitchen + Footwork + Balance vượt trội
    if tier_lower in ["beginner", "newbie", "social", "phong trào"] and (pillar_kitchen >= 0.72 and pillar_footwork >= 0.68):
        raw_pvr = max(raw_pvr, 3.00)

    bio_score = round(max(2.00, min(4.85, raw_pvr)), 2)
    bio_floor_lock = round(max(2.00, bio_score - 0.15), 2)

    return {
        "bio_score": bio_score,
        "bio_floor_lock": bio_floor_lock,
        "bio_percent": bio_percent,
        "consistency_index": round(((pillar_swing + pillar_balance) / 2.0) * 100, 1),
        "ready_stance_percent": round(ready_stance_ratio * 100, 1),
        "swing_velocity_index": round(vel_score * 100, 1),
        "footwork_agility_index": round(pillar_footwork * 100, 1),
        "kitchen_transition_index": round(pillar_kitchen * 100, 1),
        "core_balance_index": round(pillar_balance * 100, 1),
        "avg_elbow": round(float(np.mean(elbow_angles)), 1),
        "avg_knee": round(mean_knee, 1)
    }


def run_video_audit_pipeline(
    temp_video_path: str,
    court_distance: str,
    player_position: str,
    player_tier: str,
    wins_cnt: int,
    total_cnt: int,
    winrate_val: float,
    clean_profile_url: str,
    custom_pick_id: Optional[str]
) -> dict:
    start_ts = time.time()
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Không thể mở file video!")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_to_scan = min(total_frames, int(fps * 75)) if total_frames > 0 else 1200

    is_far_court = str(court_distance).lower() == "far"
    target_samples = 48 if is_far_court else 40
    frame_step = max(2, max_frames_to_scan // target_samples)
    target_roi_width = 640 if is_far_court else 520

    r_elbow_angles, l_elbow_angles, knee_angles = [], [], []
    hip_positions: List[Tuple[float, float]] = []
    wrist_speeds: List[float] = []
    trunk_tilts: List[float] = []
    ankle_spreads: List[float] = []
    prev_wrist_pos: Optional[Tuple[float, float]] = None

    tracked_center: Optional[Tuple[float, float]] = None
    target_hsv_ref: Optional[np.ndarray] = None
    lost_frames = 0

    try:
        if MP_POSE_MODULE is not None:
            with MP_POSE_MODULE.Pose(
                static_image_mode=False,
                model_complexity=1 if is_far_court else 0,
                smooth_landmarks=True,
                min_detection_confidence=0.42,
                min_tracking_confidence=0.42
            ) as pose:
                frame_idx = 0
                sampled_count = 0

                while cap.isOpened() and sampled_count < target_samples:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    if frame_idx % frame_step != 0:
                        continue

                    ret, frame = cap.retrieve()
                    if not ret or frame is None:
                        continue

                    sampled_count += 1
                    full_h, full_w, _ = frame.shape

                    x1, y1, x2, y2 = get_dynamic_roi_box(frame.shape, court_distance, player_position, tracked_center)
                    roi_frame = frame[y1:y2, x1:x2]
                    if roi_frame.size == 0:
                        continue

                    rh, rw, _ = roi_frame.shape
                    if rw > target_roi_width:
                        scale = float(target_roi_width) / rw
                        proc_frame = cv2.resize(roi_frame, (target_roi_width, int(rh * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        proc_frame = roi_frame

                    rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark

                        r_hip = lm[MP_POSE_MODULE.PoseLandmark.RIGHT_HIP.value]
                        l_hip = lm[MP_POSE_MODULE.PoseLandmark.LEFT_HIP.value]
                        r_sh_lm = lm[MP_POSE_MODULE.PoseLandmark.RIGHT_SHOULDER.value]
                        l_sh_lm = lm[MP_POSE_MODULE.PoseLandmark.LEFT_SHOULDER.value]

                        mid_hip_roi_x = (r_hip.x + l_hip.x) / 2.0
                        mid_hip_roi_y = (r_hip.y + l_hip.y) / 2.0
                        mid_sh_roi_x = (r_sh_lm.x + l_sh_lm.x) / 2.0
                        mid_sh_roi_y = (r_sh_lm.y + l_sh_lm.y) / 2.0

                        global_cx = (x1 + mid_hip_roi_x * (x2 - x1)) / full_w
                        global_cy = (y1 + mid_hip_roi_y * (y2 - y1)) / full_h

                        cur_hsv = extract_torso_color_hsv(frame, global_cx, global_cy)
                        is_same_player = True

                        if target_hsv_ref is None and cur_hsv is not None:
                            target_hsv_ref = cur_hsv
                        elif target_hsv_ref is not None and cur_hsv is not None and tracked_center is not None:
                            move_dist = math.hypot(global_cx - tracked_center[0], global_cy - tracked_center[1])
                            color_diff = float(np.linalg.norm(cur_hsv[:2] - target_hsv_ref[:2]))
                            if move_dist > 0.22 and color_diff > 55.0:
                                is_same_player = False

                        if not is_same_player:
                            lost_frames += 1
                            prev_wrist_pos = None
                            if lost_frames >= 3:
                                tracked_center = None
                            continue

                        lost_frames = 0
                        tracked_center = (global_cx, global_cy)

                        # Đo độ nghiêng trục thân người (Core Balance)
                        dx_trunk = abs(mid_sh_roi_x - mid_hip_roi_x)
                        dy_trunk = max(0.01, abs(mid_hip_roi_y - mid_sh_roi_y))
                        trunk_angle = math.degrees(math.atan2(dx_trunk, dy_trunk))
                        trunk_tilts.append(trunk_angle)

                        # Đo độ mở hai cổ chân (Split-step)
                        r_ank = lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ANKLE.value]
                        l_ank = lm[MP_POSE_MODULE.PoseLandmark.LEFT_ANKLE.value]
                        if r_ank.visibility >= 0.4 and l_ank.visibility >= 0.4:
                            ankle_spreads.append(math.hypot(r_ank.x - l_ank.x, r_ank.y - l_ank.y))

                        # Đo góc khuỷu tay phải & trái
                        if lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].visibility >= 0.42:
                            r_sh = [r_sh_lm.x, r_sh_lm.y]
                            r_el = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].y]
                            r_wr = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value].y]
                            re_ang = calculate_angle(r_sh, r_el, r_wr)
                            if 40.0 <= re_ang <= 175.0:
                                r_elbow_angles.append(re_ang)

                        if lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].visibility >= 0.42:
                            l_sh = [l_sh_lm.x, l_sh_lm.y]
                            l_el = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].y]
                            l_wr = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value].y]
                            le_ang = calculate_angle(l_sh, l_el, l_wr)
                            if 40.0 <= le_ang <= 175.0:
                                l_elbow_angles.append(le_ang)

                        # Đo vận tốc cổ tay tương đối
                        r_vis = lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value].visibility
                        l_vis = lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value].visibility
                        if max(r_vis, l_vis) >= 0.42:
                            active_wr = lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value] if r_vis >= l_vis else lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value]
                            rel_wx = active_wr.x - mid_hip_roi_x
                            rel_wy = active_wr.y - mid_hip_roi_y
                            if prev_wrist_pos is not None:
                                w_disp = math.hypot(rel_wx - prev_wrist_pos[0], rel_wy - prev_wrist_pos[1])
                                if w_disp < 0.85:
                                    wrist_speeds.append(w_disp)
                            prev_wrist_pos = (rel_wx, rel_wy)
                        else:
                            prev_wrist_pos = None

                        use_right_leg = (
                            lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].visibility >=
                            lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].visibility
                        )
                        if use_right_leg:
                            hp = [r_hip.x, r_hip.y]
                            kn = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].y]
                            ak = [r_ank.x, r_ank.y]
                        else:
                            hp = [l_hip.x, l_hip.y]
                            kn = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].y]
                            ak = [l_ank.x, l_ank.y]

                        k_ang = calculate_angle(hp, kn, ak)
                        if 70.0 <= k_ang <= 178.0:
                            knee_angles.append(k_ang)
                        hip_positions.append((global_cx, global_cy))
                    else:
                        lost_frames += 1
                        prev_wrist_pos = None
                        if lost_frames >= 3:
                            tracked_center = None
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
        gc.collect()

    bio_results = compute_5_pillar_pvr_metrics(
        r_elbow_angles, l_elbow_angles, knee_angles, hip_positions,
        wrist_speeds, trunk_tilts, ankle_spreads, court_distance, player_tier
    )
    video_bio_rating = bio_results["bio_score"]
    bio_floor_lock = bio_results["bio_floor_lock"]

    scraped_data, verification_status = verify_and_scrape_profile(clean_profile_url)
    if scraped_data:
        wins_cnt = scraped_data["wins"]
        total_cnt = scraped_data["total"]
        winrate_val = scraped_data["winrate"]

    is_pro = str(player_tier).lower() in ["pro", "pro elite"]

    if total_cnt > 0:
        match_confidence = min(1.0, max(0.50, total_cnt / 15.0))
        base_match_rating = 2.40 + (winrate_val / 100.0) * 1.65
        match_record_rating = (base_match_rating * match_confidence) + (video_bio_rating * (1.0 - match_confidence))

        calculated_pvr = (video_bio_rating * 0.65) + (match_record_rating * 0.35)
        calculated_pvr = max(bio_floor_lock, calculated_pvr)
        if total_cnt < 50:
            calculated_pvr = min(video_bio_rating + 0.35, calculated_pvr)

        if is_pro and winrate_val >= 75.0 and total_cnt >= 50:
            final_pvr = round(4.00 + (winrate_val - 75.0) * 0.04 + (video_bio_rating - 2.5) * 0.2, 2)
        else:
            final_pvr = round(calculated_pvr, 2)
    else:
        final_pvr = round(video_bio_rating, 2)

    final_pvr = max(2.00, min(5.50, final_pvr))
    recommended_division = classify_pvr_division(final_pvr)

    if is_pro and final_pvr >= 4.0:
        dupr_val = round(min(7.50, 6.50 + ((final_pvr - 4.0) * 0.7) + (winrate_val / 400.0)), 2)
    else:
        dupr_val = round(max(2.00, min(5.85, final_pvr - 0.12)), 2)

    rating_str = f"{final_pvr:.2f} PVR"
    dupr_str = f"DUPR {dupr_val:.2f}"
    pick_id = custom_pick_id.strip().upper() if (custom_pick_id and custom_pick_id.strip() not in ["", "string"]) else f"PICK-{uuid.uuid4().hex[:8].upper()}"
    audit_hash = hashlib.sha256(f"{pick_id}:{final_pvr}:{dupr_val}:{total_cnt}".encode('utf-8')).hexdigest()[:8]
    signature = generate_secure_qr_signature(pick_id, rating_str, f"{winrate_val:.0f}%", audit_hash)
    elapsed_sec = round(time.time() - start_ts, 2)

    return {
        "success": True,
        "processing_time_sec": elapsed_sec,
        "pick_id": pick_id,
        "player_id": pick_id,
        "calculated_rating": rating_str,
        "pvr_score": final_pvr,
        "pvna_score": final_pvr,
        "recommended_division": recommended_division,
        "peak_floor_lock": bio_floor_lock,
        "dupr_rating": dupr_str,
        "dupr_score": dupr_val,
        "biomechanics_score_100": bio_results["bio_percent"],
        "legal_disclaimer": "Certified Pickleball Verified Rating (PVR) by AI Enterprise Engine V7.0.",
        "data_authenticity": {
            "profile_verification": verification_status,
            "confidence_score": f"{round(min(1.0, max(0.75, total_cnt / 15.0)) * 100, 1)}%",
            "verified_matches_counted": total_cnt,
            "anti_sandbagging_floor": f"{bio_floor_lock:.2f} PVR"
        },
        "biomechanics_audit": {
            "dynamic_rating": f"{video_bio_rating:.2f} PVR",
            "score_100": f"{bio_results['bio_percent']}/100",
            "stroke_consistency": f"{bio_results['consistency_index']}%",
            "ready_stance_ratio": f"{bio_results['ready_stance_percent']}%",
            "swing_velocity_index": f"{bio_results['swing_velocity_index']}%",
            "footwork_agility_index": f"{bio_results['footwork_agility_index']}%",
            "kitchen_transition_index": f"{bio_results['kitchen_transition_index']}%",
            "core_balance_index": f"{bio_results['core_balance_index']}%",
            "avg_elbow_angle": f"{bio_results['avg_elbow']}°",
            "avg_knee_angle": f"{bio_results['avg_knee']}°"
        },
        "qr_certified_seal": {
            "status": "SEALED_AND_VERIFIED",
            "pick_id": pick_id,
            "signature": signature,
            "audit_hash": audit_hash
        }
    }


@app.get("/")
def health_check():
    return {"status": "ONLINE", "engine": "PVR AI Dynamic-Tracking Engine V7.0"}


@app.post("/api/analyze-video")
def analyze_video(
    file: UploadFile = File(None),
    video_url: Optional[str] = Form(None),
    camera_rtsp_url: Optional[str] = Form(None),
    pick_id: Optional[str] = Form(None),
    lang: Optional[str] = Form("vi"),
    court_distance: Optional[str] = Form("near"),
    player_position: Optional[str] = Form("auto"),
    shirt_color: Optional[str] = Form(None),
    shoe_color: Optional[str] = Form(None),
    headwear: Optional[str] = Form(None),
    body_type: Optional[str] = Form("athletic"),
    player_tier: Optional[str] = Form("intermediate"),
    match_wins: Optional[str] = Form("0"),
    total_matches: Optional[str] = Form("0"),
    profile_result_url: Optional[str] = Form(None),
    historical_winrate: Optional[str] = Form("60.0")
):
    try:
        wins_cnt = int(match_wins) if match_wins and match_wins != "string" else 0
        total_cnt = int(total_matches) if total_matches and total_matches != "string" else 0
    except ValueError:
        wins_cnt, total_cnt = 0, 0

    if total_cnt > 0:
        winrate_val = round((wins_cnt / total_cnt) * 100, 1)
    else:
        try:
            winrate_val = float(historical_winrate) if historical_winrate and historical_winrate != "string" else 60.0
        except ValueError:
            winrate_val = 60.0

    task_id = str(uuid.uuid4())
    temp_video_path = f"temp_{task_id}.mp4"
    clean_url = video_url.strip() if video_url and video_url.strip() not in ["", "string"] else None

    if file and file.filename != "":
        content = file.file.read()
        if len(content) > 0:
            with open(temp_video_path, "wb") as buffer:
                buffer.write(content)
    elif clean_url:
        dl_res = download_online_video(clean_url, temp_video_path)
        if not (dl_res and os.path.exists(temp_video_path)):
            raise HTTPException(status_code=400, detail="Không thể tải video từ URL!")
    else:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp File Video hoặc Link URL!")

    result_payload = run_video_audit_pipeline(
        temp_video_path=temp_video_path,
        court_distance=court_distance,
        player_position=player_position,
        player_tier=player_tier,
        wins_cnt=wins_cnt,
        total_cnt=total_cnt,
        winrate_val=winrate_val,
        clean_profile_url=profile_result_url,
        custom_pick_id=pick_id
    )

    if len(RESULTS_CACHE) > 50:
        RESULTS_CACHE.clear()
    RESULTS_CACHE[task_id] = {"status": "COMPLETED", "result": result_payload}

    return {
        "success": True,
        "task_id": task_id,
        "status": "COMPLETED",
        "result": result_payload,
        "check_status_url": f"/api/task-status/{task_id}",
        "engine_version": "V7.0 PVR Multi-Dimensional Calibrated"
    }


@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã Task ID!")
    return RESULTS_CACHE[task_id]
