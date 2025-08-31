from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import random
from datetime import datetime
import json

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Load trained model
MODEL_PATH = "Model/brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names with detailed information
class_info = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glia cells that surround nerve cells.',
        'severity': 'High',
        'color': '#FF6B6B'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor that arises from the meninges — the membranes that surround your brain and spinal cord.',
        'severity': 'Medium',
        'color': '#FFE66D'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'No signs of tumor detected in the brain MRI scan. The brain tissue appears normal.',
        'severity': 'None',
        'color': '#4ECDC4'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A tumor that forms in the pituitary gland near the brain. Most pituitary tumors are benign.',
        'severity': 'Medium',
        'color': '#45B7D1'
    }
}

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Store history of predictions
prediction_history = []

# Model statistics
model_stats = {
    'accuracy': 95.37,
    'dataset': 'Brain Tumor MRI Dataset',
    'total_images': 'Large dataset with multiple tumor types',
    'model_type': 'Convolutional Neural Network (CNN)'
}

# ---------------------------
# Utils
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(img_path):
    """Make prediction on uploaded image"""
    img = preprocess_image(img_path)
    if img is None:
        return None, 0
    
    preds = model.predict(img, verbose=0)
    class_index = np.argmax(preds)
    class_name = class_names[class_index]
    confidence = float(np.max(preds)) * 100
    
    # Get all class probabilities
    all_probs = {}
    for i, name in enumerate(class_names):
        all_probs[name] = float(preds[0][i]) * 100
    
    return class_name, confidence, all_probs

def get_file_size(filepath):
    """Get file size in human readable format"""
    size_bytes = os.path.getsize(filepath)
    if size_bytes == 0:
        return "0B"
    size_name = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", 
                                 history=prediction_history, 
                                 model_stats=model_stats,
                                 class_info=class_info,
                                 error="No file selected")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", 
                                 history=prediction_history, 
                                 model_stats=model_stats,
                                 class_info=class_info,
                                 error="No file selected")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            
            try:
                file.save(filepath)
                file_size = get_file_size(filepath)
                
                # Predict
                result = predict_image(filepath)
                if result[0] is None:
                    return render_template("index.html", 
                                         history=prediction_history, 
                                         model_stats=model_stats,
                                         class_info=class_info,
                                         error="Error processing image. Please try again.")
                
                pred_class, confidence, all_probs = result
                
                # Save to history
                prediction_entry = {
                    "filename": filename,
                    "original_filename": file.filename,
                    "class": pred_class,
                    "confidence": round(confidence, 2),
                    "file_size": file_size,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "all_probabilities": {k: round(v, 2) for k, v in all_probs.items()}
                }
                prediction_history.insert(0, prediction_entry)  # Add to beginning
                
                # Keep only last 10 predictions
                if len(prediction_history) > 10:
                    prediction_history.pop()
                
                return render_template("index.html",
                                     filename=filename,
                                     pred_class=pred_class,
                                     confidence=round(confidence, 2),
                                     all_probs={k: round(v, 2) for k, v in all_probs.items()},
                                     history=prediction_history,
                                     model_stats=model_stats,
                                     class_info=class_info,
                                     file_size=file_size,
                                     success=True)
            
            except Exception as e:
                return render_template("index.html", 
                                     history=prediction_history, 
                                     model_stats=model_stats,
                                     class_info=class_info,
                                     error=f"Error processing file: {str(e)}")
        else:
            return render_template("index.html", 
                                 history=prediction_history, 
                                 model_stats=model_stats,
                                 class_info=class_info,
                                 error="Invalid file type. Please upload PNG, JPG, or JPEG images only.")
    
    return render_template("index.html", 
                         history=prediction_history, 
                         model_stats=model_stats,
                         class_info=class_info)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

@app.route("/api/stats")
def api_stats():
    """API endpoint for model statistics"""
    return jsonify({
        'model_stats': model_stats,
        'class_info': class_info,
        'total_predictions': len(prediction_history)
    })

@app.route("/clear_history")
def clear_history():
    """Clear prediction history"""
    global prediction_history
    prediction_history = []
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", 
                         history=prediction_history, 
                         model_stats=model_stats,
                         class_info=class_info,
                         error="File too large! Please upload files smaller than 16MB."), 413

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", 
                         history=prediction_history, 
                         model_stats=model_stats,
                         class_info=class_info,
                         error="Page not found."), 404

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)