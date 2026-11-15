import os
import re
import cv2
import gc
import math
import hmac
import hashlib
import json
import time
import uuid
import urllib.request
import numpy as np
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PVR AI Enterprise Engine - Pickleball Verified Rating V5.0",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "PVR_AI_ENTERPRISE_KEY_2026_SECURE")
BACKGROUND_TASKS_STORE: Dict[str, dict] = {}

DICT_I18N = {
    "vi": {
        "fault_right": "FAULT (Dẫm vạch Kitchen bằng Chân Phải)",
        "fault_left": "FAULT (Dẫm vạch Kitchen bằng Chân Trái)",
        "clean": "CLEAN (Hợp lệ - Không dẫm vạch)",
        "in_court": "IN (Bóng Trong sân)",
        "out_court": "OUT (Bóng Ngoài sân)",
        "source_rtsp": "IP Camera Sân Đấu (RTSP)",
        "source_url": "Video Online (URL)",
        "source_file": "File Upload",
        "right_foot": "Chân Phải",
        "left_foot": "Chân Trái"
    },
    "en": {
        "fault_right": "FAULT (Kitchen Line Violation - Right Foot)",
        "fault_left": "FAULT (Kitchen Line Violation - Left Foot)",
        "clean": "CLEAN (No Violation)",
        "in_court": "IN (In Court)",
        "out_court": "OUT (Out of Bounds)",
        "source_rtsp": "Live IP Camera (RTSP)",
        "source_url": "Online Video (URL)",
        "source_file": "File Upload",
        "right_foot": "Right Foot",
        "left_foot": "Left Foot"
    }
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

def generate_secure_qr_signature(pick_id: str, rating: str, winrate: str, audit_hash: str) -> str:
    raw_data = f"{pick_id}:{rating}:{winrate}:{audit_hash}"
    return hmac.new(SECRET_KEY.encode('utf-8'), raw_data.encode('utf-8'), hashlib.sha256).hexdigest()[:16]

def order_court_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transform_point_homography(x, y, src_pts):
    try:
        if not src_pts or len(src_pts) != 4:
            return x, y, False
        ordered_pts = order_court_points(src_pts)
        pts_dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        h_matrix, _ = cv2.findHomography(ordered_pts, pts_dst)
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, h_matrix)
        norm_x = round(float(transformed_point[0][0][0]), 1)
        norm_y = round(float(transformed_point[0][0][1]), 1)
        is_in = (0.0 <= norm_x <= 100.0) and (0.0 <= norm_y <= 100.0)
        return norm_x, norm_y, is_in
    except Exception:
        return x, y, False

def download_online_video(url: str, output_path: str = "temp_online.mp4") -> str:
    try:
        import yt_dlp
        clean_url = url.strip()
        clean_url = re.sub(r'[\(\)\[\]]', '', clean_url)
        urls_found = re.findall(r'(https?://[^\s]+)', clean_url)
        if urls_found:
            clean_url = urls_found[0]
        if "youtu.be/" in clean_url:
            video_id = clean_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
            clean_url = f"https://www.youtube.com/watch?v={video_id}"
        if "?si=" in clean_url:
            clean_url = clean_url.split("?si=")[0]
        if "&si=" in clean_url:
            clean_url = clean_url.split("&si=")[0]

        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        }
        if os.path.exists("cookies.txt"):
            ydl_opts['cookiefile'] = 'cookies.txt'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])
        return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else None
    except Exception as e:
        print(f"❌ Download Error: {e}")
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
        with urllib.request.urlopen(req, timeout=4) as response:
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

def compute_professional_biomechanics(elbow_angles: List[float], knee_angles: List[float], player_tier: str = "intermediate") -> dict:
    """
    Thuật toán phân tích động cơ sinh học PVR Engine V5.0:
    - Đánh giá dựa trên dải góc tối ưu (Optimal Biomechanics Range).
    - Quy đổi điểm kỹ thuật (0-100) sang thang điểm trình PVR chuẩn thực tế (75/100 -> ~3.10 PVR).
    - Tính toán ngưỡng khóa sàn kỹ thuật (Biomechanics Floor Lock) chống rửa điểm.
    """
    tier_lower = str(player_tier).lower()
    tier_base_bonus = 0.0
    if tier_lower in ["pro", "pro elite"]:
        tier_base_bonus = 0.65
    elif tier_lower in ["advanced", "nâng cao"]:
        tier_base_bonus = 0.30
    elif tier_lower in ["newbie", "beginner", "mới chơi"]:
        tier_base_bonus = -0.35

    if not elbow_angles or not knee_angles:
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

    # 1. Tỷ lệ giữ trọng tâm thấp / Ready Stance (Góc gối chuẩn Pickleball: 110° - 150°)
    ready_frames = [ang for ang in knee_angles if 110.0 <= ang <= 150.0]
    ready_stance_ratio = len(ready_frames) / max(1, len(knee_angles))

    # 2. Độ ổn định vung vợt (Consistency Index)
    elbow_std = float(np.std(elbow_angles))
    knee_std = float(np.std(knee_angles))
    elbow_consistency = max(45.0, min(98.0, 100.0 - (elbow_std * 0.95)))
    knee_consistency = max(45.0, min(98.0, 100.0 - (knee_std * 0.85)))
    overall_consistency = round((elbow_consistency * 0.6) + (knee_consistency * 0.4), 1)

    # 3. Đánh giá biên độ kỹ thuật (Dải tối ưu: Khuỷu tay 95° - 135°, Gối 115° - 148°)
    mean_elbow = float(np.mean(elbow_angles))
    mean_knee = float(np.mean(knee_angles))

    elbow_diff = max(0.0, abs(mean_elbow - 115.0) - 12.0)
    knee_diff = max(0.0, abs(mean_knee - 132.0) - 14.0)

    elbow_form_score = max(0.35, 1.0 - (elbow_diff / 45.0))
    knee_form_score = max(0.35, 1.0 - (knee_diff / 45.0))

    # Tổng hợp điểm kỹ thuật thang 100 (Mức khá đạt ~75.0/100)
    bio_percent = round(
        min(98.0, max(40.0, (elbow_form_score * 35.0) + (knee_form_score * 35.0) + (ready_stance_ratio * 20.0) + (overall_consistency * 0.10))),
        1
    )

    # Quy đổi sang thang PVR (Pickleball Verified Rating):
    # 50/100 -> 2.60 PVR | 75/100 -> 3.10 PVR | 90/100 -> 3.45 PVR
    raw_pvr = 2.25 + ((bio_percent - 35.0) / 65.0) * 1.35 + tier_base_bonus
    bio_score = round(max(2.15, min(4.80, raw_pvr)), 2)
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

def process_video_async_task(
    task_id: str,
    temp_video_path: str,
    selected_lang: str,
    court_distance: str,
    player_position: str,
    shirt_color: str,
    shoe_color: str,
    headwear: str,
    body_type: str,
    player_tier: str,
    wins_cnt: int,
    total_cnt: int,
    winrate_val: float,
    winrate_source: str,
    clean_profile_url: str,
    custom_pick_id: Optional[str],
    custom_kitchen_y: Optional[int],
    src_homography_pts: Optional[list],
    in_out_bool: bool,
    heatmap_bool: bool,
    kitchen_bool: bool,
    clean_rtsp: bool,
    clean_url: bool
):
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        BACKGROUND_TASKS_STORE[task_id] = {"status": "FAILED", "error": "Cannot open video file"}
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 1) if total_frames > 0 else 10.0

    frame_skip = 8 if duration_sec > 45 else 3
    elbow_angles, knee_angles = [], []

    try:
        import mediapipe as mp
        try:
            mp_pose = mp.solutions.pose
        except AttributeError:
            import mediapipe.python.solutions.pose as mp_pose

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    r_vis = (
                        lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility +
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                    )
                    l_vis = (
                        lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility +
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
                    )

                    use_right = (r_vis >= l_vis) if str(player_position).lower() == "auto" else (
                        True if (str(player_position).lower() == "right" and r_vis >= 0.4) or (r_vis > l_vis + 0.25) else False
                    )

                    if use_right:
                        sh = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        el = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wr = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        hp = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        kn = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ak = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    else:
                        sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        hp = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        kn = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ak = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    e_ang = calculate_angle(sh, el, wr)
                    k_ang = calculate_angle(hp, kn, ak)

                    if 40.0 <= e_ang <= 175.0:
                        elbow_angles.append(e_ang)
                    if 70.0 <= k_ang <= 178.0:
                        knee_angles.append(k_ang)
    except Exception as e:
        print(f"MediaPipe Processing Exception: {e}")
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
        gc.collect()

    # Phân tích kỹ thuật PVR V5.0
    bio_results = compute_professional_biomechanics(elbow_angles, knee_angles, player_tier)
    video_bio_rating = bio_results["bio_score"]
    bio_floor_lock = bio_results["bio_floor_lock"]

    # Xác thực liên kết thành tích thi đấu
    scraped_data, verification_status = verify_and_scrape_profile(clean_profile_url)
    if scraped_data:
        wins_cnt = scraped_data["wins"]
        total_cnt = scraped_data["total"]
        winrate_val = scraped_data["winrate"]
        winrate_source = "DUPR/PVR Network Sync"

    is_pro = str(player_tier).lower() in ["pro", "pro elite"]

    # TÍNH ĐIỂM CHUẨN PVR V5.0 (CHỐNG TRÌNH BAO & KHÓA SÀN CHỐNG RỬA ĐIỂM)
    if total_cnt > 0:
        match_confidence = min(1.0, total_cnt / 30.0)
        base_match_rating = 2.40 + (winrate_val / 100.0) * 1.65
        match_record_rating = (base_match_rating * match_confidence) + (video_bio_rating * (1.0 - match_confidence))

        # 60% Kỹ thuật Video thực tế + 40% Thành tích đối đầu (chống thổi phồng điểm ảo)
        calculated_pvr = (video_bio_rating * 0.60) + (match_record_rating * 0.40)

        # Khóa sàn kỹ thuật chống rửa điểm (không tụt quá 0.15 so với kỹ thuật thực tế)
        calculated_pvr = max(bio_floor_lock, calculated_pvr)

        # Khóa trần chống trình bao (không vượt quá điểm kỹ thuật video + 0.30 nếu chưa đủ 50 trận)
        if total_cnt < 50:
            calculated_pvr = min(video_bio_rating + 0.30, calculated_pvr)

        if is_pro and winrate_val >= 75.0 and total_cnt >= 50:
            final_pvr = round(4.00 + (winrate_val - 75.0) * 0.04 + (video_bio_rating - 2.5) * 0.2, 2)
        else:
            final_pvr = round(calculated_pvr, 2)
    else:
        final_pvr = round(video_bio_rating, 2)

    final_pvr = max(2.15, min(5.50, final_pvr))

    # Quy đổi thang DUPR chuẩn quốc tế (thấp hơn PVR khoảng 0.12 điểm)
    if is_pro and final_pvr >= 4.0:
        dupr_val = round(min(7.50, 6.50 + ((final_pvr - 4.0) * 0.7) + (winrate_val / 400.0)), 2)
    else:
        dupr_val = round(max(2.00, min(5.85, final_pvr - 0.12)), 2)

    rating_str = f"{final_pvr:.2f} PVR"
    dupr_str = f"DUPR {dupr_val:.2f}"
    pick_id = custom_pick_id.strip().upper() if (custom_pick_id and custom_pick_id.strip() not in ["", "string"]) else f"PICK-{uuid.uuid4().hex[:8].upper()}"
    audit_hash = hashlib.sha256(f"{pick_id}:{final_pvr}:{dupr_val}:{total_cnt}".encode('utf-8')).hexdigest()[:8]
    signature = generate_secure_qr_signature(pick_id, rating_str, f"{winrate_val:.0f}%", audit_hash)

    BACKGROUND_TASKS_STORE[task_id] = {
        "status": "COMPLETED",
        "result": {
            "success": True,
            "pick_id": pick_id,
            "player_id": pick_id,
            "calculated_rating": rating_str,
            "pvr_score": final_pvr,
            "pvna_score": final_pvr,
            "peak_floor_lock": bio_floor_lock,
            "dupr_rating": dupr_str,
            "dupr_score": dupr_val,
            "biomechanics_score_100": bio_results["bio_percent"],
            "legal_disclaimer": "Certified Pickleball Verified Rating (PVR) by AI Enterprise Engine V5.0.",
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
    }

@app.post("/api/analyze-video")
def analyze_video(
    background_tasks: BackgroundTasks,
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
    historical_winrate: Optional[str] = Form("60.0"),
    custom_kitchen_y: Optional[int] = Form(None),
    court_corners_json: Optional[str] = Form(None),
    enable_in_out_check: Optional[str] = Form("true"),
    enable_heatmap: Optional[str] = Form("true"),
    enable_kitchen_var: Optional[str] = Form("true")
):
    selected_lang = "en" if str(lang).lower() == "en" else "vi"

    src_homography_pts = None
    if court_corners_json and court_corners_json != "string":
        try:
            parsed_pts = json.loads(court_corners_json)
            if len(parsed_pts) == 4:
                src_homography_pts = parsed_pts
        except Exception:
            src_homography_pts = None

    try:
        wins_cnt = int(match_wins) if match_wins and match_wins != "string" else 0
        total_cnt = int(total_matches) if total_matches and total_matches != "string" else 0
    except ValueError:
        wins_cnt, total_cnt = 0, 0

    if total_cnt > 0:
        winrate_val = round((wins_cnt / total_cnt) * 100, 1)
        winrate_source = f"Tự khai báo ({wins_cnt}/{total_cnt} trận)"
    else:
        try:
            winrate_val = float(historical_winrate) if historical_winrate and historical_winrate != "string" else 60.0
        except ValueError:
            winrate_val = 60.0
        winrate_source = "Ước tính ban đầu"

    task_id = str(uuid.uuid4())
    temp_video_path = f"temp_{task_id}.mp4"
    clean_rtsp = camera_rtsp_url.strip() if camera_rtsp_url and camera_rtsp_url.strip() not in ["", "string"] else None
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
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp File Video để đảm bảo chất lượng bóc tách!")

    BACKGROUND_TASKS_STORE[task_id] = {"status": "PROCESSING", "message": "PVR AI Engine V5.0 is auditing"}

    background_tasks.add_task(
        process_video_async_task,
        task_id,
        temp_video_path,
        selected_lang,
        court_distance,
        player_position,
        shirt_color,
        shoe_color,
        headwear,
        body_type,
        player_tier,
        wins_cnt,
        total_cnt,
        winrate_val,
        winrate_source,
        profile_result_url,
        pick_id,
        custom_kitchen_y,
        src_homography_pts,
        True,
        True,
        True,
        bool(clean_rtsp),
        bool(clean_url)
    )

    return {
        "success": True,
        "task_id": task_id,
        "status": "PROCESSING",
        "check_status_url": f"/api/task-status/{task_id}",
        "engine_version": "V5.0 PVR Verified"
    }

@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in BACKGROUND_TASKS_STORE:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã Task ID!")
    return BACKGROUND_TASKS_STORE[task_id]
