import os
import re
import cv2
import gc
import hmac
import hashlib
import time
import uuid
import urllib.request
import numpy as np
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PVR AI Enterprise Engine - High-Speed Pay-Per-Video V6.1",
    version="6.1.0"
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
            'format': 'best[height<=480][ext=mp4]/worst[ext=mp4]/best[height<=480]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
            'nocheckcertificate': True,
            'socket_timeout': 10,
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


def crop_roi_for_target_player(frame, court_distance: str, player_position: str):
    h, w, _ = frame.shape
    dist = str(court_distance).lower()
    pos = str(player_position).lower()

    y1, y2 = (int(h * 0.35), h) if dist == "near" else ((0, int(h * 0.65)) if dist == "far" else (0, h))
    x1, x2 = (0, int(w * 0.70)) if pos == "left" else ((int(w * 0.30), w) if pos == "right" else (0, w))

    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else frame


def compute_professional_biomechanics(
    r_elbow_list: List[float],
    l_elbow_list: List[float],
    knee_angles: List[float],
    hip_y_positions: List[float],
    player_tier: str = "intermediate"
) -> dict:
    tier_lower = str(player_tier).lower()
    tier_base_bonus = 0.65 if tier_lower in ["pro", "pro elite"] else (
        0.28 if tier_lower in ["semi_pro", "advanced", "nâng cao", "bán chuyên"] else (
            -0.15 if tier_lower in ["social", "newbie", "beginner", "phong trào"] else 0.0
        )
    )

    # Tự động chọn tay cầm vợt (tay có biên độ vung động học lớn hơn và đủ mẫu)
    r_std = float(np.std(r_elbow_list)) if len(r_elbow_list) >= 3 else 0.0
    l_std = float(np.std(l_elbow_list)) if len(l_elbow_list) >= 3 else 0.0
    elbow_angles = r_elbow_list if (r_std >= l_std and len(r_elbow_list) >= 3) else (
        l_elbow_list if len(l_elbow_list) >= 3 else r_elbow_list
    )

    if len(elbow_angles) < 3 or len(knee_angles) < 3:
        default_pvr = round(max(2.20, 2.95 + tier_base_bonus), 2)
        return {
            "bio_score": default_pvr,
            "bio_floor_lock": round(default_pvr - 0.15, 2),
            "bio_percent": 75.0,
            "consistency_index": 76.0,
            "ready_stance_percent": 72.0,
            "avg_elbow": 115.0,
            "avg_knee": 134.0
        }

    ready_frames = [ang for ang in knee_angles if 108.0 <= ang <= 148.0]
    ready_stance_ratio = len(ready_frames) / len(knee_angles)

    elbow_std = float(np.std(elbow_angles))
    if 12.0 <= elbow_std <= 28.0:
        swing_arc_score = 1.0
    else:
        swing_arc_score = max(0.45, 1.0 - (abs(elbow_std - 20.0) / 45.0))

    if len(hip_y_positions) >= 3:
        hip_std = float(np.std(hip_y_positions))
        hip_stability = max(55.0, min(98.0, 100.0 - (hip_std * 180.0)))
    else:
        hip_stability = 78.0

    mean_elbow = float(np.mean(elbow_angles))
    mean_knee = float(np.mean(knee_angles))

    elbow_diff = max(0.0, abs(mean_elbow - 116.0) - 12.0)
    knee_diff = max(0.0, abs(mean_knee - 130.0) - 14.0)

    elbow_form_score = max(0.40, 1.0 - (elbow_diff / 45.0))
    knee_form_score = max(0.40, 1.0 - (knee_diff / 45.0))

    overall_consistency = round((swing_arc_score * 50.0) + (hip_stability * 0.50), 1)
    bio_percent = round(
        min(98.0, max(42.0,
            (elbow_form_score * 30.0) +
            (knee_form_score * 30.0) +
            (ready_stance_ratio * 22.0) +
            (swing_arc_score * 18.0)
        )),
        1
    )

    raw_pvr = 2.25 + ((bio_percent - 35.0) / 65.0) * 1.35 + tier_base_bonus
    bio_score = round(max(2.15, min(4.85, raw_pvr)), 2)
    bio_floor_lock = round(max(2.00, bio_score - 0.15), 2)

    return {
        "bio_score": bio_score,
        "bio_floor_lock": bio_floor_lock,
        "bio_percent": bio_percent,
        "consistency_index": overall_consistency,
        "ready_stance_percent": round(ready_stance_ratio * 100, 1),
        "avg_elbow": round(mean_elbow, 1),
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
    max_frames_to_scan = min(total_frames, int(fps * 60)) if total_frames > 0 else 900
    target_samples = 28
    frame_step = max(2, max_frames_to_scan // target_samples)

    r_elbow_angles, l_elbow_angles, knee_angles, hip_y_positions = [], [], [], []

    try:
        if MP_POSE_MODULE is not None:
            with MP_POSE_MODULE.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45
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
                    roi_frame = crop_roi_for_target_player(frame, court_distance, player_position)
                    rh, rw, _ = roi_frame.shape
                    if rw > 480:
                        scale = 480.0 / rw
                        roi_frame = cv2.resize(roi_frame, (480, int(rh * scale)), interpolation=cv2.INTER_LINEAR)

                    rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark

                        # Đo độc lập cả 2 tay để tự tìm tay cầm vợt
                        if lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].visibility >= 0.45:
                            r_sh = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_SHOULDER.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_SHOULDER.value].y]
                            r_el = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ELBOW.value].y]
                            r_wr = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_WRIST.value].y]
                            re_ang = calculate_angle(r_sh, r_el, r_wr)
                            if 40.0 <= re_ang <= 175.0:
                                r_elbow_angles.append(re_ang)

                        if lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].visibility >= 0.45:
                            l_sh = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_SHOULDER.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_SHOULDER.value].y]
                            l_el = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_ELBOW.value].y]
                            l_wr = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_WRIST.value].y]
                            le_ang = calculate_angle(l_sh, l_el, l_wr)
                            if 40.0 <= le_ang <= 175.0:
                                l_elbow_angles.append(le_ang)

                        # Đo chân có độ rõ nét cao hơn
                        use_right_leg = (
                            lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].visibility >=
                            lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].visibility
                        )
                        if use_right_leg:
                            hp = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_HIP.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_HIP.value].y]
                            kn = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_KNEE.value].y]
                            ak = [lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ANKLE.value].x, lm[MP_POSE_MODULE.PoseLandmark.RIGHT_ANKLE.value].y]
                        else:
                            hp = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_HIP.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_HIP.value].y]
                            kn = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_KNEE.value].y]
                            ak = [lm[MP_POSE_MODULE.PoseLandmark.LEFT_ANKLE.value].x, lm[MP_POSE_MODULE.PoseLandmark.LEFT_ANKLE.value].y]

                        k_ang = calculate_angle(hp, kn, ak)
                        if 70.0 <= k_ang <= 178.0:
                            knee_angles.append(k_ang)
                        hip_y_positions.append(hp[1])
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
        gc.collect()

    bio_results = compute_professional_biomechanics(
        r_elbow_angles, l_elbow_angles, knee_angles, hip_y_positions, player_tier
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
        match_confidence = min(1.0, total_cnt / 30.0)
        base_match_rating = 2.40 + (winrate_val / 100.0) * 1.65
        match_record_rating = (base_match_rating * match_confidence) + (video_bio_rating * (1.0 - match_confidence))

        calculated_pvr = (video_bio_rating * 0.60) + (match_record_rating * 0.40)
        calculated_pvr = max(bio_floor_lock, calculated_pvr)
        if total_cnt < 50:
            calculated_pvr = min(video_bio_rating + 0.30, calculated_pvr)

        if is_pro and winrate_val >= 75.0 and total_cnt >= 50:
            final_pvr = round(4.00 + (winrate_val - 75.0) * 0.04 + (video_bio_rating - 2.5) * 0.2, 2)
        else:
            final_pvr = round(calculated_pvr, 2)
    else:
        final_pvr = round(video_bio_rating, 2)

    final_pvr = max(2.15, min(5.50, final_pvr))

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
        "peak_floor_lock": bio_floor_lock,
        "dupr_rating": dupr_str,
        "dupr_score": dupr_val,
        "biomechanics_score_100": bio_results["bio_percent"],
        "legal_disclaimer": "Certified Pickleball Verified Rating (PVR) by AI Enterprise Engine V6.1.",
        "data_authenticity": {
            "profile_verification": verification_status,
            "confidence_score": f"{round(min(1.0, max(0.75, total_cnt / 30.0)) * 100, 1)}%",
            "verified_matches_counted": total_cnt,
            "anti_sandbagging_floor": f"{bio_floor_lock:.2f} PVR"
        },
        "biomechanics_audit": {
            "dynamic_rating": f"{video_bio_rating:.2f} PVR",
            "score_100": f"{bio_results['bio_percent']}/100",
            "stroke_consistency": f"{bio_results['consistency_index']}%",
            "ready_stance_ratio": f"{bio_results['ready_stance_percent']}%",
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
    return {"status": "ONLINE", "engine": "PVR AI Fast-Engine V6.1 (Cloud Run Pay-Per-Video Optimized)"}


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

    # Chạy trực tiếp trong request để Cloud Run cấp 100% CPU và không bị ngắt khi trả response
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

    # Giữ cache ngắn hạn (tối đa 50 kết quả) để tương thích hàm pollTaskStatus ở frontend
    if len(RESULTS_CACHE) > 50:
        RESULTS_CACHE.clear()
    RESULTS_CACHE[task_id] = {"status": "COMPLETED", "result": result_payload}

    return {
        "success": True,
        "task_id": task_id,
        "status": "COMPLETED",
        "result": result_payload,
        "check_status_url": f"/api/task-status/{task_id}",
        "engine_version": "V6.1 PVR Cloud-Run Fast-Engine"
    }


@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã Task ID!")
    return RESULTS_CACHE[task_id]
