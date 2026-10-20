import os
import torch
from PIL import Image
import numpy as np
from config import DEVICE, MODEL_SAVE_PATH, COGNITIVE_LOAD_CLASSES
from model import ACLENetCNN

def predict_and_explain(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Gambar '{image_path}' tidak ditemukan.")
        return
        
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.tensor(img_np, dtype=torch.float32)
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    model = ACLENetCNN(num_classes=len(COGNITIVE_LOAD_CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probabilities)
        pred_class = COGNITIVE_LOAD_CLASSES[pred_idx]
        confidence = probabilities[pred_idx] * 100

    print("==================================================")
    print("      ACLE-Net HCI & UI EVALUATION RESULT        ")
    print("==================================================")
    print(f"Input Gambar       : {image_path}")
    print(f"Hasil Prediksi     : {pred_class.upper()}")
    print(f"Tingkat Keyakinan  : {confidence:.2f}%\n")
    
    print("Rekomendasi Desain Antarmuka (Berdasarkan Prinsip IMK):")
    if pred_class == "high_load":
        print("⚠️ REVISI DIPERLUKAN (Extraneous Cognitive Load Tinggi):")
        print(" 1. Otak manusia hanya mampu menyimpan memori jangka pendek 7 ± 2 item.")
        print("    Antarmuka ini terlalu padat visual. Sederhanakan tata letak (low-fidelity prototyping).")
        print(" 2. Terapkan Shneiderman's Mantra: 'Overview first, zoom & filter, details-on-demand'.")
        print(" 3. Hilangkan Chartjunk (elemen 3D/dekorasi tak bermakna - Edward Tufte).")
        print(" 4. Pastikan pemrosesan Preattentive Visual Attributes (warna, ukuran) terarah (<200ms).")
    elif pred_class == "medium_load":
        print("⚡ OPTIMALISASI MODERAT:")
        print(" 1. Kelompokkan informasi terkait agar tidak membebani working memory.")
        print(" 2. Periksa standar kontras WCAG untuk aksesibilitas lansia & pengguna disabilitas.")
    else:
        print("✅ DESAIN BAGUS (Inklusif & Beban Kognitif Rendah):")
        print(" 1. Antarmuka efisien dan mengurangi beban mental pengguna.")
        print(" 2. Filosofi User-Centered Design (UCD) terpenuhi dengan baik.")
    print("==================================================")

if __name__ == "__main__":
    test_img = "data/cognitive_load/high_load/ui_high_0.png"
    predict_and_explain(test_img)
