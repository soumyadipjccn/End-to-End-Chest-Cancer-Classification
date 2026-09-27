import os
import sys
sys.path.extend(["src", "."])
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from chestCancerClassifier.utils.common import decodeImage
from chestCancerClassifier.pipeline.prediction import PredictionPipeline
from chestCancerClassifier import logger
import main

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = None

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/health", methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({"status": "healthy", "service": "Chest Cancer Classification AI API", "version": "1.0.0"}), 200

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train_route():
    try:
        def start_training():
            main.run_pipeline()
        
        # Run training in thread so server doesn't block
        thread = threading.Thread(target=start_training)
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training pipeline initiated in background. Check running_logs.log for detailed progress."
        }), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    try:
        global clApp
        if clApp is None:
            clApp = ClientApp()

        image_data = None
        
        # Check if file upload or base64 JSON payload
        if 'image' in request.files:
            file = request.files['image']
            file_path = os.path.join("static", "uploads", "uploaded_scan.jpg")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            clApp.classifier = PredictionPipeline(file_path)
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            file_path = "inputImage.jpg"
            decodeImage(image_data, file_path)
            clApp.classifier = PredictionPipeline(file_path)
        elif request.json and 'sample_path' in request.json:
            file_path = request.json['sample_path']
            clApp.classifier = PredictionPipeline(file_path)
        else:
            return jsonify({"error": "No image or sample_path provided"}), 400

        result = clApp.classifier.predict()
        return jsonify(result)

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
