import streamlit as st
import torch #lib pythorch buat yolov8
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image #proses gambar

# Atur layout halaman
st.set_page_config(layout="wide", page_title="Deteksi Objek dengan YOLOv8")

# Tambahkan CSS: padding di semua sisi 
# CSS minimal padding + kecilkan judul
st.markdown("""
    <style>
        .block-container {
            padding-top: 0px;
            padding-bottom: 5px;
            padding-left: 30px;
            padding-right: 30px;
        }
        h1 {
            font-size: 2.5rem;
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True) #memperbolehkan penggunaan html/css scr lgsg


# Load model
import os
import urllib.request

MODEL_URL = "https://drive.google.com/file/d/1srOW1ub1-o42P-3CkV2D0CZy_5odIHt3/view?usp=sharing"
MODEL_PATH = "sken4.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = YOLO(MODEL_PATH)

# Warna tiap kelas
color_map = {
    "arabika": (0, 0, 139),
    "liberika": (128, 0, 128),
    "robusta": (255, 0, 0)
}

st.title("Deteksi Objek dengan YOLOv8")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Konversi ke BGR untuk OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Jalankan deteksi dg YOLO
    results = model(img_bgr, conf=0.5)

    #Kasih id di tiap objek
    class_counter = {"arabika": 0, "liberika": 0, "robusta": 0}

    #Salinan gambar untuk diberi bounding box
    img_annotated = img_np.copy()

    # Loop untuk setiap bounding box yang terdeteksi
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #Ambil koordinat objek dan informasi kelas
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Rename label jika perlu
            if class_name == "a":
                class_name = "arabika"
            elif class_name == "l":
                class_name = "liberika"
            elif class_name == "r":
                class_name = "robusta"

            #Hitung bounding box
            class_counter[class_name] += 1
            obj_id = class_counter[class_name]

            color = color_map.get(class_name, (0, 255, 0))

            # Gambar bounding box dan label
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_annotated, f"{class_name} id:{obj_id} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Hitung total objek
    total_objects = sum(class_counter.values())

    # Hitung persentase dan siapkan statistik sebagai list markdown
    percentages = {}
    report_lines = []
    for cls_name, count in class_counter.items():
        if total_objects > 0:
            percentages[cls_name] = (count / total_objects) * 100
        else:
            percentages[cls_name] = 0
        report_lines.append(f"<li><b>{cls_name.capitalize()}</b>: {count} objek ({percentages[cls_name]:.1f}%)</li>")

     # Siapkan statistik deteksi sebagai HTML
    report_html = f"""
        <div style="background-color: #1e1e1e; border-radius: 15px; padding: 25px; margin-top: 40px;
                    box-shadow: 0 0 10px rgba(255,255,255,0.1);">
            <h3 style="color: #FFD700;">📊 Statistik Deteksi</h3>
            <p style="font-size: 18px; color: white;"><b>Total objek terdeteksi:</b> {total_objects}</p>
            <ul style="font-size: 17px; color: white; list-style-type: disc; margin-left: 20px;">
                {''.join(report_lines)}
            </ul>
        </div>
    """

    # Kolom: gambar asli | spacer | hasil deteksi | spacer | statistik
    spacer_left, col1, col2, col3, spacer_right = st.columns([0.2, 1, 1, 1.1, 0.2])

    with col1:
        st.image(img_np, caption="📷 Gambar Asli", use_column_width=True)

    with col2:
        st.image(img_annotated, caption="📌 Hasil Deteksi", use_column_width=True)

    with col3:
        st.markdown(report_html, unsafe_allow_html=True)

