from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
from src.utils import prepare_features_from_raw


app = Flask(__name__)
CORS(app)  # Allow React to talk to Flask

# Load models and metrics
try:
    MODELS = {
        "lr": joblib.load("models/lr_food_model.joblib"),
        "rf": joblib.load("models/rf_food_model.joblib"),
        "gbr": joblib.load("models/gbr_food_model.joblib"),
    }
    with open("models/model_metrics.json") as f:
        MODEL_METRICS = json.load(f)
except FileNotFoundError:
    print("FATAL ERROR: Models or metrics missing. Did you run src/preprocessing.py and src/model.py?")
    MODELS = {}
    MODEL_METRICS = {}


@app.route("/", methods=["GET"])
def home():
    """Simple API welcome."""
    return jsonify({
        "message": "Somalia Food Price Predictor API. Check /metrics and use POST /predict.",
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns the pre-calculated performance metrics for the frontend."""
    if not MODEL_METRICS:
        return jsonify({"error": "Metrics unavailable. Please train models."}), 503
    return jsonify(MODEL_METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts 5 core input features and returns a price prediction."""

    # Check model choice
    choice = (request.args.get("model") or "").lower()
    valid_models = list(MODELS.keys())
    if choice not in MODELS:
        return jsonify({"error": f"Model not found. Use model={('|').join(valid_models)}"}), 400
    model = MODELS[choice]

    data = request.get_json(silent=True) or {}

    # ONLY 5 features required from the user
    required = ["commodity", "market", "unit", "year", "month"]
    missing = [k for k in required if k not in data]

    if missing:
        return jsonify({"error": f"Need these 5 features: {', '.join(missing)}"}), 400

    try:
        # Prepare features, utils.py injects the default values for 'hidden' features
        x_new = prepare_features_from_raw(data)
        pred = float(model.predict(x_new)[0])
    except Exception as e:
        # A nice error message if something breaks during prediction
        print(f"Prediction logic error: {e}")
        return jsonify({"error": "Failed to run prediction. Check that the input format matches the expected features."}), 500

    # Return the result
    return jsonify({
        "model": choice,
        "predicted_price": round(pred, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
