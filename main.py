import os
import re
import cv2
import math
import hmac
import hashlib
import json
import time
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Pickleball AI Enterprise Engine - Tournament & VAR Controller",
    version="2.8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "PICKLEBALL_AI_SECRET_KEY_SUPER_SECURE_2026"

# Bộ nhớ tạm lưu trữ các trận đấu đang diễn ra
MATCH_SESSIONS = {}

DICT_I18N = {
    "vi": {
        "fault_right": "FAULT (Dẫm vạch Kitchen bằng Chân Phải)",
        "fault_left": "FAULT (Dẫm vạch Kitchen bằng Chân Trái)",
        "clean": "CLEAN (Hợp lệ - Không dẫm vạch)",
        "in_court": "IN (Bóng Trong sân)",
        "out_court": "OUT (Bóng Ngoài sân)",
        "right_foot": "Chân Phải",
        "left_foot": "Chân Trái"
    },
    "en": {
        "fault_right": "FAULT (Kitchen Line Violation - Right Foot)",
        "fault_left": "FAULT (Kitchen Line Violation - Left Foot)",
        "clean": "CLEAN (No Violation)",
        "in_court": "IN (In Court)",
        "out_court": "OUT (Out of Bounds)",
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

def generate_secure_qr_signature(player_id: str, rating: str, winrate: str) -> str:
    raw_data = f"{player_id}:{rating}:{winrate}"
    return hmac.new(SECRET_KEY.encode('utf-8'), raw_data.encode('utf-8'), hashlib.sha256).hexdigest()[:12]

def capture_rtsp_stream(rtsp_url: str, output_path: str = "temp_rtsp.mp4", duration_sec: int = 5) -> bool:
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        max_frames = fps * duration_sec

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_recorded = 0
        while cap.isOpened() and frames_recorded < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_recorded += 1

        cap.release()
        out.release()
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"❌ RTSP Error: {e}")
        return False

def detect_ball_center(frame, lower_yellow, upper_yellow):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if 3 < radius < 25:
            return (int(x), int(y))
    return None

@app.get("/")
def root():
    return {"status": "Active", "system": "Pickleball AI Enterprise Engine - Tournament Ready V2.8.0"}

# --- 🎯 API BỔ SUNG CHO BAN TỔ CHỨC VÀ TRỌNG TÀI ---

@app.post("/api/tournament/start-match")
def start_match(
    match_id: str = Form(..., description="Mã trận đấu, ví dụ: MATCH-101"),
    court_number: str = Form("Court 1", description="Tên sân"),
    player_a: str = Form("Đội A"),
    player_b: str = Form("Đội B"),
    rtsp_url: str = Form(..., description="Link Camera IP hoặc App điện thoại RTSP của sân")
):
    """BTC khởi tạo trận đấu và gán Camera cho Trọng tài"""
    MATCH_SESSIONS[match_id] = {
        "court": court_number,
        "player_a": player_a,
        "player_b": player_b,
        "rtsp_url": rtsp_url,
        "start_time": time.strftime("%H:%M:%S - %d/%m/%Y"),
        "var_logs": []
    }
    return {
        "success": True,
        "message": f"Đã khởi tạo trận {match_id} tại {court_number}",
        "match_info": MATCH_SESSIONS[match_id]
    }

@app.post("/api/tournament/referee-check-var")
def referee_check_var(
    match_id: str = Form(..., description="Mã trận đấu đang diễn ra"),
    lang: Optional[str] = Form("vi")
):
    """Trọng tài bấm 1-Click để kiểm tra VAR 5 giây gần nhất từ Camera sân"""
    if match_id not in MATCH_SESSIONS:
        raise HTTPException(status_code=404, detail="Không tìm thấy trận đấu! Vui lòng nhờ BTC khởi tạo trận.")

    session = MATCH_SESSIONS[match_id]
    rtsp_url = session["rtsp_url"]
    temp_path = f"temp_var_{match_id}.mp4"

    # Trích xuất 5 giây từ Camera IP
    success = capture_rtsp_stream(rtsp_url, temp_path, duration_sec=5)
    if not success:
        raise HTTPException(status_code=400, detail="Không thể kết nối với Camera IP sân đấu!")

    # Chạy AI kiểm tra nhanh
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    kitchen_line_y = int(height * 0.45)
    fault_detected = False
    violating_foot = ""
    selected_lang = "en" if str(lang).lower() == "en" else "vi"
    t = DICT_I18N[selected_lang]

    try:
        import mediapipe as mp
        try:
            mp_pose = mp.solutions.pose
        except AttributeError:
            import mediapipe.python.solutions.pose as mp_pose

        with mp_pose.Pose(static_image_mode=False, model_complexity=0) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    r_foot_y = int(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * height)
                    l_foot_y = int(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * height)

                    if r_foot_y <= kitchen_line_y + 5:
                        fault_detected = True
                        violating_foot = t["fault_right"]
                        break
                    elif l_foot_y <= kitchen_line_y + 5:
                        fault_detected = True
                        violating_foot = t["fault_left"]
                        break
    except Exception as e:
        print(f"VAR Error: {e}")
    finally:
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    decision = violating_foot if fault_detected else t["clean"]
    log_entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "decision": decision,
        "status": "FAULT" if fault_detected else "CLEAN"
    }
    MATCH_SESSIONS[match_id]["var_logs"].append(log_entry)

    return {
        "success": True,
        "match_id": match_id,
        "referee_decision": decision,
        "status": "FOOT_FAULT" if fault_detected else "NO_FAULT",
        "timestamp": log_entry["timestamp"]
    }

@app.get("/api/tournament/match-summary/{match_id}")
def get_match_summary(match_id: str):
    """BTC xem lại nhật ký VAR toàn bộ trận đấu"""
    if match_id not in MATCH_SESSIONS:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu trận đấu!")
    return MATCH_SESSIONS[match_id]
