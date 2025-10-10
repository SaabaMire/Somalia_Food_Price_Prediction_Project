
# Somalia Food Price Prediction Project

## 📜 Project Description

This is a complete, full-stack Machine Learning solution for predicting highly **volatile food commodity prices** in Somalia. The predictive system features a multi-model Python backend (Flask API) and a modern **React.js + Tailwind CSS** frontend.

**Key Feature: Smart Input Defaults.** The solution requires the user to input **only 5 core, easy-to-obtain features** (commodity, market, unit, year, month). The backend intelligently enriches the request by automatically providing the **6 necessary geospatial and administrative features** (e.g., admin1, latitude) using pre-calculated defaults, ensuring a fast, user-friendly experience without sacrificing model accuracy.

## ✅ Core Requirements

| Requirement | Status | Notes |
| :--- | :--- | :--- |
| **Simplified User Input** | **DONE** | API requires only 5 key features; 6 remaining features are handled by smart defaults. |
| **Frontend Tech** | **DONE** | Implemented with React.js and Tailwind CSS for a beautiful, responsive UI. |
| **Algorithms** | **DONE** | Includes Linear Regression (LR), Random Forest (RF), and **Gradient Boosting Regressor (GBR)**. |
| **Deployment** | **DONE** | Flask API exposes /predict and /metrics endpoints for dynamic prediction. |
| **Documentation** | **DONE** | Comprehensive README.md and detailed project_paper.md provided. |

---

## 🚀 Setup & Run Instructions

### 1. Backend Setup (Python)

Ensure you have Python 3.8+ installed and all dependencies listed in `requirements.txt` are installed (`pip install -r requirements.txt`).

**Note:** Ensure your file structure includes a `dataset/` directory containing `wfp_food_prices_som.csv`, `src/` for training scripts, and `api/` for the server, as per project standards.

```bash
# 1. Clean data and generate preprocessing artifacts (scaler, columns list, and defaults)
python src/preprocessing.py

# 2. Train and evaluate all three models (LR, RF, GBR) and save them to the models/ directory
python src/model.py

# 3. Start the API server
python api/app.py
# The API will be available at [http://127.0.0.1:8080](http://127.0.0.1:8080)
````

### 2\. Frontend Setup (React/JS)

The frontend is a single React component file (`frontend/src/PredictionForm.js`) designed to interact with the Python API running on port 8080.

1.  **Serve the React Component:** In a production setup, you would use a tool like Vite or webpack. For this environment, simply ensure the file is served, pointing API calls to `http://127.0.0.1:8080`.

2.  **Interaction:** The frontend handles user input, model selection, and displays the predicted price and model performance metrics fetched from the `/predict` and `/metrics` API endpoints, respectively.

-----

## 3\. Example API Usage (Inference)

The API provides two endpoints: `/metrics` (GET) and `/predict` (POST).

### 3.1 Fetching Model Metrics

```bash
curl -X GET "[http://127.0.0.1:8080/metrics](http://127.0.0.1:8080/metrics)"
```

**Example Response:**

```json
{
  "gbr": { "R2": 0.895, "MAE": 3519.0 },
  "lr": { "R2": 0.807, "MAE": 5194.0 },
  "rf": { "R2": 0.981, "MAE": 1049.0 }
}
```

### 3.2 Making a Prediction (Using 5-Feature Input)

The backend handles the missing `admin1`, `latitude`, etc., based on the provided `market` and `commodity` through smart lookup logic defined in `utils.py`.

```bash
curl -X POST "[http://127.0.0.1:8080/predict?model=rf](http://127.0.0.1:8080/predict?model=rf)" \
-H "Content-Type: application/json" \
-d '{
    "commodity": "Sorghum (red)",
    "market": "Bakaara",
    "unit": "KG",
    "year": 2026,
    "month": 5
}'
```

**Example Prediction Response:**

```json
{
  "model": "rf",
  "predicted_price": 75000.56,
  "unit": "SLS"
}
```

-----

## 4\. Documentation and Code

  * **Project Paper:** The detailed project report and reflection paper are available in `project_paper.md`.
  * **Code Structure:**
      * `src/preprocessing.py`: Handles data cleaning, scaling, and saves necessary artifacts.
      * `src/model.py`: Trains and evaluates the three ML algorithms.
      * `api/app.py`: The Flask server defining API routes.
      * `api/utils.py`: Contains the critical `prepare_features_from_raw` function for robust inference.
