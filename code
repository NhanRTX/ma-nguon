import gradio as gr
import cv2
import numpy as np
import os
import json
import uuid
from datetime import datetime
import tensorflow as tf
from ultralytics import YOLO

os.makedirs("invoices", exist_ok=True)

detector = YOLO("/content/YOLOv8.pt")
cnn_interpreter = tf.lite.Interpreter(model_path="/content/CNNmodel.tflite")
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()

FOOD_LABELS = [
    "Ca hu kho",
    "Canh cai",
    "Canh chua",
    "Com trang",
    "Dau hu sot ca",
    "Ga chien",
    "Rau muong xao",
    "Thit kho",
    "Thit kho trung",
    "Trung chien",
]

PRICE_LIST = {
    "Ca hu kho": 22000,
    "Canh cai": 9000,
    "Canh chua": 10000,
    "Com trang": 5000,
    "Dau hu sot ca": 16000,
    "Ga chien": 25000,
    "Rau muong xao": 8000,
    "Thit kho": 17000,
    "Thit kho trung": 18000,
    "Trung chien": 12000,
}

def generate_invoice_id():
    now_str = datetime.now().strftime("%Y%m%d")
    unique_part = str(uuid.uuid4())[:8].upper()
    return f"INV{now_str}-{unique_part}"

def detect_and_classify(img):
    yolo_result = detector(img)
    bboxes = yolo_result[0].boxes.data.cpu().numpy()
    detected_dishes = []
    total_cost = 0
    for box in bboxes:
        xmin, ymin, xmax, ymax, conf, yolo_cls = box
        if conf < 0.3:
            continue
        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        if crop_img.size == 0:
            continue
        crop_resized = cv2.resize(crop_img, (224, 224))
        input_arr = np.expand_dims(crop_resized.astype(np.float32) / 255.0, axis=0)
        cnn_interpreter.set_tensor(cnn_input[0]['index'], input_arr)
        cnn_interpreter.invoke()
        preds = cnn_interpreter.get_tensor(cnn_output[0]['index'])
        dish_idx = int(np.argmax(preds))
        dish_name = FOOD_LABELS[dish_idx]
        detected_dishes.append(dish_name)
        total_cost += PRICE_LIST.get(dish_name, 0)

    if not detected_dishes:
        return "Không nhận diện được món nào!", "0đ", None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    invoice_id = generate_invoice_id()
    detail_lines = [f"- {name}: {PRICE_LIST[name]:,}đ" for name in detected_dishes]
    summary = "\n".join(detail_lines)
    total_str = f"{total_cost:,}đ"

    invoice_text = f"""HÓA ĐƠN: {invoice_id}
Thời gian: {now}
---------------------------
{summary}
---------------------------
TỔNG CỘNG: {total_str}
"""
    invoice_filename = f"invoices/{invoice_id}.txt"
    with open(invoice_filename, "w", encoding="utf-8") as f:
        f.write(invoice_text)

    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    else:
        history_data = []

    history_data.append({
        "id": invoice_id,
        "time": now,
        "items": detected_dishes,
        "total": total_cost,
        "file": invoice_filename
    })

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    return summary, total_str, invoice_filename

def capture_from_camera(camera_url):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        return None, "Không thể kết nối tới camera. Kiểm tra lại URL/IP."
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Không lấy được khung hình từ camera."
    return frame, "Đã chụp ảnh từ camera thành công."

def view_history():
    if not os.path.exists("history.json"):
        return "Chưa có hóa đơn nào."
    with open("history.json", "r", encoding="utf-8") as f:
        history_data = json.load(f)
    last_5 = history_data[-5:]
    lines = []
    for h in last_5:
        lines.append(f"{h['id']} | {h['time']} | {', '.join(h['items'])} | {h['total']:,}đ")
    return "\n".join(lines)

def query_invoice_by_id(invoice_id):
    invoice_id = invoice_id.strip().upper()
    if not os.path.exists("history.json"):
        return "Không có dữ liệu lịch sử."
    with open("history.json", "r", encoding="utf-8") as f:
        history_data = json.load(f)
    for h in history_data:
        if h["id"] == invoice_id:
            with open(h["file"], "r", encoding="utf-8") as f:
                return f.read()
    return f"Không tìm thấy hóa đơn với mã: {invoice_id}"

with gr.Blocks(theme=gr.themes.Soft()) as app:
    with gr.Row():
        gr.Markdown("<h2>Hệ thống nhận diện và tính tiền</h2>")

    with gr.Row():
        inp_img = gr.Image(type="numpy", label="Ảnh khay cơm")
        with gr.Column():
            out_dishes = gr.Textbox(label="Danh sách món & giá", lines=6, interactive=False)
            out_sum = gr.Textbox(label="Tổng tiền", interactive=False)
            invoice_file = gr.File(label="Tải hóa đơn", interactive=False)

    with gr.Row():
        run_btn = gr.Button("Nhận diện và xuất hóa đơn")
        view_history_btn = gr.Button("Xem 5 hóa đơn gần nhất")

    with gr.Row():
        cam_input = gr.Textbox(label="Nhập URL/IP camera (RTSP/HTTP)")
        cam_btn = gr.Button("Chụp ảnh từ camera")
        cam_status = gr.Textbox(label="Trạng thái kết nối", interactive=False)

    captured_img = gr.Image(label="Ảnh từ camera", type="numpy")

    with gr.Row():
        invoice_query = gr.Textbox(label="Nhập mã hóa đơn để tra cứu")
        query_btn = gr.Button("Tra cứu hóa đơn")

    history_output = gr.Textbox(label="Kết quả lịch sử hoặc tra cứu", lines=10, interactive=False)

    run_btn.click(fn=detect_and_classify, inputs=inp_img, outputs=[out_dishes, out_sum, invoice_file])
    view_history_btn.click(fn=view_history, outputs=history_output)
    cam_btn.click(fn=capture_from_camera, inputs=cam_input, outputs=[captured_img, cam_status])
    query_btn.click(fn=query_invoice_by_id, inputs=invoice_query, outputs=history_output)

app.launch()
