import os
import io
import base64
import zipfile
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

# -------------------
# إعدادات عامة
# -------------------
IMG_SIZE = (240, 240)
MOLE_MIN = float(os.getenv("MOLE_MIN", 0.99))
BIN_THR  = float(os.getenv("BIN_THR", 0.5))
CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
MODEL_PATH = "SkinMeleo_model.keras"  # اسم المودل بعد فك الـ ZIP
ZIP_PATH   = "SkinMeleo_model.zip"    # إذا رفعت ZIP

# -------------------
# فك ضغط ZIP إذا موجود
# -------------------
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")  # يفك الملفات في نفس المجلد
    print("✅ ZIP extracted: SkinMeleo_model.keras should now be available.")

# -------------------
# التأكد من وجود المودل
# -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# -------------------
# تحميل المودل
# -------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------
# دوال مساعدة
# -------------------
def preprocess_pil(pil_img):
    # تحويل الصورة دائماً لـ RGB
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr

def decode_b64(s):
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s)))

# -------------------
# إعداد Flask
# -------------------
app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        b64 = data.get("image_b64")
        if not b64:
            return jsonify({"error": "image_b64 is required"}), 400

        x = preprocess_pil(decode_b64(b64))
        p_mole, p_bin, p_type = model.predict(x, verbose=0)

        has_mole = float(p_mole[0][0]) >= MOLE_MIN
        img_ok = "T" if has_mole else "F"
        if not has_mole:
            return jsonify({"IsAbnormal":"F","Image_Suitability":img_ok,"Predicted_Type":""})

        is_abn = "T" if float(p_bin[0][0]) > BIN_THR else "F"
        pred = CLASS_NAMES[int(np.argmax(p_type[0]))]
        return jsonify({"IsAbnormal":is_abn,"Image_Suitability":img_ok,"Predicted_Type":pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# لتشغيل محلي أو على Render
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Render يستخدم هذا المنفذ
    app.run(host="0.0.0.0", port=port)


