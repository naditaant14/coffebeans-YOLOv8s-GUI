import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Atur layout halaman
st.set_page_config(layout="wide", page_title="Deteksi Objek dengan YOLOv8")

# Tambahkan CSS: padding kecil di semua sisi (10px-30px)
st.markdown("""
    <style>
        .block-container {
            padding-top: 60px;
            padding-bottom: 60px;
            padding-left: 60px;
            padding-right: 60px;
        }
        h1 {
            font-size: 4rem;  /* judul jadi lebih besar */
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# Load model
MODEL_PATH = "D:/SEMESTER 7/SKRIPSI/streamlit/sken 4.pt"
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

    # Jalankan deteksi
    results = model(img_bgr, conf=0.6)

    class_counter = {"arabika": 0, "liberika": 0, "robusta": 0}
    img_annotated = img_np.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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

            class_counter[class_name] += 1
            obj_id = class_counter[class_name]
            color = color_map.get(class_name, (0, 255, 0))

            # Gambar bounding box dan label
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_annotated, f"{class_name} id:{obj_id} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Tampilkan gambar berdampingan
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Gambar Asli", use_column_width=True)
    with col2:
        st.image(img_annotated, caption="Hasil Deteksi", use_column_width=True)
