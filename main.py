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
        zip_ref.extractall(".")
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
def decode_b64(b64_string):
    """تحويل Base64 إلى صورة PIL"""
    import base64, io
    from PIL import Image

    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 3)[1]
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # تأكدي من هذا السطر
    return image


def preprocess_pil(pil_img):
    """تهيئة الصورة للمودل"""
    img = pil_img.convert("RGB").resize(IMG_SIZE)  # 🔥 نضمن 3 قنوات دائمًا
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


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

        img = decode_b64(b64)
        x = preprocess_pil(img)

        # 🔥 التأكد من أن المودل يعطي 3 مخرجات
        preds = model.predict(x, verbose=0)
        if len(preds) == 3:
            p_mole, p_bin, p_type = preds
        else:
            return jsonify({"error": "Unexpected model outputs"}), 500

        has_mole = float(p_mole[0][0]) >= MOLE_MIN
        img_ok = "T" if has_mole else "F"

        if not has_mole:
            return jsonify({
                "IsAbnormal": "F",
                "Image_Suitability": img_ok,
                "Predicted_Type": ""
            })

        is_abn = "T" if float(p_bin[0][0]) > BIN_THR else "F"
        pred = CLASS_NAMES[int(np.argmax(p_type[0]))]

        return jsonify({
            "IsAbnormal": is_abn,
            "Image_Suitability": img_ok,
            "Predicted_Type": pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# لتشغيل محلي أو على Render
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)




