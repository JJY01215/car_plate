from flask import Flask, request, abort, Response, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import threading
import time
import torch
from ultralytics import YOLO
import easyocr
import pandas as pd


app = Flask(__name__)
authorized_plates_df = pd.read_csv("car.csv")
# 載入模型
license_plate_detector = YOLO('Find_license_plate.pt')
vehicle_detector = YOLO('yolov8n.pt')

# 初始化 EasyOCR
reader = easyocr.Reader(['en'])

def detect_plate(image):
    # 1. 先用 YOLOv8 檢測是否有車
    vehicle_results = vehicle_detector(image)
    vehicles = [box for box in vehicle_results[0].boxes if box.cls == 2]  # 假設 2 是車輛的類別
    
    if not vehicles:
        return None
    
    # 2. 在車輛區域中尋找車牌
    plate_results = license_plate_detector(image)
    if not plate_results[0].boxes:
        return None
        
    # 3. 獲取車牌區域model訓練利用yolov8
    plate_box = plate_results[0].boxes[0]
    x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
    plate_img = image[y1:y2, x1:x2]
    def check_authorization(plate_number):
        match = authorized_plates_df[authorized_plates_df['plate_number'] == plate_number]
        if not match.empty:
            owner = match.iloc[0]['owner']
            return f"已授權車牌，車主：{owner}"
        else:
            return "未授權車牌"
    # 4. 使用 EasyOCR 辨識車牌文字
    results = reader.readtext(plate_img)
    
    # 5. 取得辨識結果
    if results:
        # 取得最高信心度的文字
        text = results[0][1]
        # 移除空白字元
        plate_number = ''.join(text.split())
        return plate_number
    
    return None

load_dotenv()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")


line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    if plate_number:
        auth_result = check_authorization(plate_number)
        reply_message = f"偵測到的車牌號碼為：{plate_number}\n{auth_result}"

    # 將圖片保存為臨時文件
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
    
    # 讀取圖片並進行車牌辨識
    image = cv2.imread(image_path)
    plate_number = detect_plate(image)  # 使用您的車牌辨識模型
    
    # 回傳辨識結果
    if plate_number:
        reply_message = f"偵測到的車牌號碼為：{plate_number}"
    else:
        reply_message = "無法辨識車牌，請確保圖片清晰度並重試。"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_message)
    )

# 新增全域變數用於存儲最新的車牌辨識結果
latest_plate = None
camera = None

def init_camera():
    global camera
    camera = cv2.VideoCapture(0)  # 使用預設攝影機
    
def process_video_stream():
    global latest_plate, camera
    while True:
        if camera is None:
            continue
        ret, frame = camera.read()
        if not ret:
            continue
            
        # 進行車牌辨識
        try:
            plate_number = detect_plate(frame)
            if plate_number:
                latest_plate = plate_number
        except Exception as e:
            print(f"辨識錯誤: {str(e)}")
            
        time.sleep(0.1)

def gen_frames():
    global camera
    while True:
        if camera is None:
            continue
        ret, frame = camera.read()
        if not ret:
            continue
            
        # 在畫面上顯示最新辨識結果
        if latest_plate:
            cv2.putText(frame, f"Plate: {latest_plate}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    global latest_plate
    if event.message.text == "取得車牌":
        if latest_plate:
            auth_result = check_authorization(latest_plate)
            reply_message = f"目前偵測到的車牌號碼為：{latest_plate}\n{auth_result}"

        else:
            reply_message = "目前未偵測到任何車牌"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_message)
        )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    init_camera()
    # 啟動視訊處理線程
    video_thread = threading.Thread(target=process_video_stream, daemon=True)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)