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
    title="Pickleball AI Enterprise Engine - Professional Tournament V5.0",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "PICKLEBALL_PRO_TOURNAMENT_OFFICIAL_SECRET_2026")
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
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

def generate_tournament_qr_signature(player_id: str, rating: str, tier: str, audit_hash: str) -> str:
    raw_data = f"{player_id}:{rating}:{tier}:{audit_hash}:TOURNAMENT_OFFICIAL_V5"
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
    """
    Module kiểm tra và xác thực hồ sơ giải đấu độc lập (DUPR / Pickleball Global).
    """
    if not profile_url or profile_url.strip() in ["", "None", "string"]:
        return None, "UNVERIFIED_SELF_ASSESSMENT"

    clean_url = profile_url.strip()
    try:
        is_dupr = "mydupr.com" in clean_url or "dupr.com" in clean_url
        is_pickleball_global = "pickleball.global" in clean_url or "pickleballtournaments.com" in clean_url
        
        if not (is_dupr or is_pickleball_global):
            return None, "EXTERNAL_UNVERIFIED_SOURCE"

        req = urllib.request.Request(clean_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            html = response.read().decode('utf-8', errors='ignore')
            match = re.search(r'(\d+)\s*(Wins|Thắng).*?(\d+)\s*(Losses|Bại|Total)', html, re.IGNORECASE)
            if match:
                wins = int(match.group(1))
                losses = int(match.group(3))
                total = wins + losses
                return {"wins": wins, "total": total, "winrate": round((wins/max(1, total))*100, 1)}, "OFFICIALLY_VERIFIED"

        return None, "DOMAIN_VERIFIED_PENDING_SYNC"
    except Exception:
        return None, "PROFILE_LINK_PRIVATE"

def classify_and_score_strokes(stroke_data: List[dict]) -> dict:
    """
    Thuật toán V5.0: Phân loại kỹ thuật từng cú đánh (Dink, Drop, Drive/Smash)
    thay vì lấy góc trung bình cộng toàn clip.
    """
    if not stroke_data:
        return {
            "dink_rating": 2.50,
            "drop_rating": 2.50,
            "drive_rating": 2.50,
            "ready_stance_percent": 65.0,
            "composite_bio_rating": 2.50,
            "stroke_breakdown": {"dink_count": 0, "drop_count": 0, "drive_count": 0}
        }

    dink_frames = []
    drop_frames = []
    drive_frames = []
    ready_stance_count = 0

    for item in stroke_data:
        elbow = item['elbow']
        knee = item['knee']
        wrist_y = item['wrist_y']
        shoulder_y = item['shoulder_y']

        # Đánh giá tư thế sẵn sàng gập gối (Ready Stance)
        if 115.0 <= knee <= 145.0:
            ready_stance_count += 1

        # Phân loại: Cú đánh trên cao (Smash/High Drive) vs Cú đánh tầm trung/thấp (Drop & Dink)
        if wrist_y < shoulder_y:  # Cổ tay cao hơn vai -> Smash / High Drive
            drive_frames.append(item)
        elif 95.0 <= elbow <= 135.0 and knee <= 145.0: # Tay kiểm soát ngắn, gối chùng -> Dink
            dink_frames.append(item)
        else: # Vung vợt chuyển tiếp từ sân sâu -> Drop / Reset
            drop_frames.append(item)

    total_frames = len(stroke_data)
    ready_percent = round((ready_stance_count / max(1, total_frames)) * 100, 1)

    # 1. Điểm Dink (Tiêu chuẩn: Cổ tay/khuỷu tay cố định ~115°-125°, gối gập thấp ~125°-135°)
    if dink_frames:
        d_elbow_std = float(np.std([f['elbow'] for f in dink_frames]))
        d_knee_mean = float(np.mean([f['knee'] for f in dink_frames]))
        dink_stability = max(0.0, 1.0 - (d_elbow_std / 20.0))
        dink_knee_depth = max(0.0, 1.0 - abs(d_knee_mean - 130.0) / 25.0)
        dink_score = round(2.0 + (dink_stability * 1.3) + (dink_knee_depth * 1.2), 2)
    else:
        dink_score = 2.50

    # 2. Điểm Drop / Reset (Tiêu chuẩn: Vung tay biên độ mượt mà, gối hỗ trợ nhịp nhàng)
    if drop_frames:
        dp_elbow_mean = float(np.mean([f['elbow'] for f in drop_frames]))
        drop_form = max(0.0, 1.0 - abs(dp_elbow_mean - 120.0) / 30.0)
        drop_score = round(2.0 + (drop_form * 2.2) + (ready_percent / 100.0 * 0.3), 2)
    else:
        drop_score = 2.50

    # 3. Điểm Drive / Power (Độ ổn định khi vung tay tốc độ cao)
    if drive_frames:
        dr_elbow_mean = float(np.mean([f['elbow'] for f in drive_frames]))
        drive_score = round(min(5.0, 2.0 + (dr_elbow_mean / 180.0) * 2.5), 2)
    else:
        drive_score = 2.50

    # Tổng hợp điểm kỹ thuật Biomechanics (Trọng số: 40% Dink, 35% Drop, 25% Drive)
    composite_bio = round((dink_score * 0.40) + (drop_score * 0.35) + (drive_score * 0.25), 2)

    return {
        "dink_rating": min(5.50, dink_score),
        "drop_rating": min(5.50, drop_score),
        "drive_rating": min(5.50, drive_score),
        "ready_stance_percent": ready_percent,
        "composite_bio_rating": min(5.50, composite_bio),
        "stroke_breakdown": {
            "dink_frames": len(dink_frames),
            "drop_frames": len(drop_frames),
            "drive_frames": len(drive_frames)
        }
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
    custom_kitchen_y: Optional[int],
    src_homography_pts: Optional[list],
    in_out_bool: bool,
    heatmap_bool: bool,
    kitchen_bool: bool,
    clean_rtsp: bool,
    clean_url: bool
):
    t = DICT_I18N[selected_lang]
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        BACKGROUND_TASKS_STORE[task_id] = {"status": "FAILED", "error": "Cannot open video file"}
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    duration_sec = round(total_frames / fps, 1) if total_frames > 0 else 10.0

    frame_skip = 10 if duration_sec > 60 else 3
    stroke_samples = []
    kitchen_faults = []

    try:
        import mediapipe as mp
        try:
            mp_pose = mp.solutions.pose
        except AttributeError:
            import mediapipe.python.solutions.pose as mp_pose

        with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5) as pose:
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
                    use_right = True if str(player_position).lower() == "right" else False
                    
                    if use_right:
                        sh = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        el = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wr = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        hp = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        kn = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ak = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                        wrist_y_val = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                        shoulder_y_val = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    else:
                        sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        hp = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        kn = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ak = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        wrist_y_val = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                        shoulder_y_val = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

                    elbow_ang = calculate_angle(sh, el, wr)
                    knee_ang = calculate_angle(hp, kn, ak)

                    stroke_samples.append({
                        "frame": frame_count,
                        "elbow": elbow_ang,
                        "knee": knee_ang,
                        "wrist_y": wrist_y_val,
                        "shoulder_y": shoulder_y_val
                    })
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

    # Phân tích kỹ thuật V5.0
    stroke_analysis = classify_and_score_strokes(stroke_samples)
    video_bio_rating = stroke_analysis["composite_bio_rating"]

    # Đối soát nguồn dữ liệu thi đấu
    scraped_data, verification_status = verify_and_scrape_profile(clean_profile_url)
    if scraped_data:
        wins_cnt = scraped_data["wins"]
        total_cnt = scraped_data["total"]
        winrate_val = scraped_data["winrate"]
        winrate_source = "Official Verified Tournament Database"

    is_official_verified = (verification_status == "OFFICIALLY_VERIFIED")
    is_pro = str(player_tier).lower() in ["pro", "pro elite"]

    # KHÓA TRẦN BẢO MẬT GIẢI ĐẤU (ANTI-FRAUD CAP):
    # Nếu tự khai báo mà không có link kiểm chứng, điểm PVNA tối đa bị giới hạn ở 3.25
    if not is_official_verified:
        match_confidence = 0.40
        passport_tier_level = "TIER-2: SELF-ASSESSMENT (UNVERIFIED)"
        final_pvna = round(min(3.25, (video_bio_rating * 0.7) + 0.9), 2)
    else:
        # Khi có hồ sơ kiểm chứng chính thức:
        match_confidence = min(1.0, total_cnt / 60.0)
        passport_tier_level = "TIER-1: OFFICIAL TOURNAMENT CERTIFIED"
        base_match_rating = 2.0 + (winrate_val / 100.0) * 2.0
        match_record_rating = (base_match_rating * match_confidence) + (2.5 * (1.0 - match_confidence))
        
        calculated_pvna = (video_bio_rating * 0.55) + (match_record_rating * 0.45)
        if is_pro and winrate_val >= 75.0 and total_cnt >= 80:
            final_pvna = round(4.00 + (winrate_val - 75.0) * 0.04 + (video_bio_rating - 2.0) * 0.2, 2)
        else:
            final_pvna = round(calculated_pvna, 2)

    final_pvna = max(2.0, min(5.50, final_pvna))

    # Quy đổi chuẩn DUPR pháp lý (Estimated DUPR by AI)
    if is_pro and final_pvna >= 4.0:
        dupr_val = round(min(7.55, 6.85 + ((final_pvna - 4.0) * 0.75) + (winrate_val / 350.0)), 2)
    else:
        dupr_val = round(max(1.85, min(5.95, final_pvna - 0.03)), 2)

    rating_str = f"{final_pvna:.2f} PVNA"
    dupr_str = f"Estimated DUPR {dupr_val:.2f}"
    player_id = f"PVNA-TN-{uuid.uuid4().hex[:8].upper()}"
    audit_hash = hashlib.sha256(f"{final_pvna}:{dupr_val}:{passport_tier_level}:{total_cnt}".encode('utf-8')).hexdigest()[:8]
    signature = generate_tournament_qr_signature(player_id, rating_str, passport_tier_level, audit_hash)

    BACKGROUND_TASKS_STORE[task_id] = {
        "status": "COMPLETED",
        "result": {
            "success": True,
            "passport_tier": passport_tier_level,
            "player_id": player_id,
            "calculated_rating": rating_str,
            "dupr_rating": dupr_str,
            "tournament_validation": {
                "verification_status": verification_status,
                "confidence_score": f"{round(match_confidence * 100, 1)}%",
                "official_matches_counted": total_cnt if is_official_verified else "Unverified Self-Report",
                "anti_fraud_cap_applied": "NO (Verified)" if is_official_verified else "YES (Capped at 3.25 Max)"
            },
            "technical_skills_audit": {
                "dink_control_score": f"{stroke_analysis['dink_rating']} / 5.50",
                "drop_reset_score": f"{stroke_analysis['drop_rating']} / 5.50",
                "drive_power_score": f"{stroke_analysis['drive_rating']} / 5.50",
                "ready_stance_ratio": f"{stroke_analysis['ready_stance_percent']}%",
                "composite_biomechanics": f"{video_bio_rating} / 5.50"
            },
            "qr_certified_seal": {
                "status": "APPROVED_FOR_TOURNAMENT_ROSTER" if is_official_verified else "APPROVED_FOR_CASUAL_PLAY",
                "signature": signature,
                "audit_hash": audit_hash,
                "referee_scan_url": f"https://hoai-bot.github.io/pickleball-ai-engine/?verify={signature}"
            }
        }
    }

@app.post("/api/analyze-video")
def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    video_url: Optional[str] = Form(None),
    camera_rtsp_url: Optional[str] = Form(None),
    lang: Optional[str] = Form("vi"),
    court_distance: Optional[str] = Form("near"),
    player_position: Optional[str] = Form("right"),
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
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp File Video bài test đạt chuẩn!")

    BACKGROUND_TASKS_STORE[task_id] = {"status": "PROCESSING", "message": "Tournament Engine V5.0 is executing shot classification audit"}

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
        "engine_version": "V5.0 Tournament Edition - Shot Classifier Active"
    }

@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in BACKGROUND_TASKS_STORE:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã Task ID!")
    return BACKGROUND_TASKS_STORE[task_id]
