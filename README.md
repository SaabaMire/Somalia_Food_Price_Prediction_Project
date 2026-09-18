# Somalia Food Price Predictor

A full-stack machine-learning application for estimating food commodity prices in Somalia. The project combines a Flask prediction API, three regression models, and a React/Vite interface.

[View the live application](https://somalia-food-price-prediction-project-m04z3xyfw-isfahan.vercel.app/) · [Open the prediction API](https://web-production-0bd50.up.railway.app/)

## Overview

The application asks for five inputs:

- Commodity
- Market
- Unit
- Year
- Month

The backend enriches these inputs with market location and commodity metadata stored in lookup files. It then prepares the model features and returns a price estimate in Somali shillings (SLS).

## Models

The training pipeline compares three regression algorithms:

| Model | R² | MAE (SLS) | RMSE (SLS) |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.8074 | 5,194.45 | 7,043.28 |
| Random Forest | 0.9808 | 1,048.87 | 2,224.20 |
| Gradient Boosting | 0.8954 | 3,518.89 | 5,190.07 |

These values come from the included `models/model_metrics.json` artifact. Random Forest produced the strongest test-set result in the current experiment.

## Tech Stack

- Python, pandas, NumPy, and scikit-learn
- Flask and Flask-CORS
- React 19, Vite, and Tailwind CSS
- Joblib for saved preprocessing and model artifacts

## Repository Structure

```text
.
├── api/                 # Flask API
├── dataset/             # Raw and processed Somalia food-price data
├── frontend/            # React/Vite user interface
├── models/              # Trained models, metrics, and lookup artifacts
├── src/                 # Preprocessing, training, and inference utilities
├── project_paper.md     # Detailed project report
└── requirements.txt     # Python dependencies
```

## Getting Started

Run all backend commands from the repository root because the scripts use root-relative paths.

### 1. Clone the repository

```bash
git clone https://github.com/SaabaMire/Somalia_Food_Price_Prediction_Project.git
cd Somalia_Food_Price_Prediction_Project
```

### 2. Set up the Python environment

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the data and train the models

```bash
python src/preprocessing.py
python src/model.py
```

These commands regenerate the cleaned dataset, lookup maps, scaler, metrics, and all three trained model files.

### 4. Start the Flask API

```bash
python api/app.py
```

The API will be available at `http://127.0.0.1:8000`.

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

On macOS or Linux, use `cp .env.example .env.local` instead of `copy`.
Open the local URL printed by Vite. `VITE_API_BASE_URL` tells the frontend where to find the Flask API.

## API Usage

### Health check

```bash
curl http://127.0.0.1:8000/
```

### Model metrics

```bash
curl http://127.0.0.1:8000/metrics
```

### Price prediction

Choose `lr`, `rf`, or `gbr` with the `model` query parameter:

```bash
curl -X POST "http://127.0.0.1:8000/predict?model=rf" \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "Sorghum (red)",
    "market": "Bakaara",
    "unit": "KG",
    "year": 2026,
    "month": 5
  }'
```

Example response:

```json
{
  "model": "rf",
  "predicted_price": 75000.56
}
```

## How Prediction Works

1. `src/preprocessing.py` cleans the dataset, builds categorical features, scales numeric fields, and creates lookup artifacts.
2. `src/model.py` trains and evaluates Linear Regression, Random Forest, and Gradient Boosting models.
3. `src/utils.py` enriches the five user inputs with stored market and commodity defaults and reproduces the training feature layout.
4. `api/app.py` exposes the saved models through `/predict` and reports evaluation results through `/metrics`.

## Documentation

See [`project_paper.md`](project_paper.md) for the project report and methodology.

## Deployment

Deploy the backend first, then give its public URL to the frontend.

### Railway backend

1. Create a Railway project from this GitHub repository.
2. Keep the root directory set to the repository root.
3. Railway installs `requirements.txt` and uses the included `Procfile`:

   ```text
   web: gunicorn --bind 0.0.0.0:$PORT api.app:app
   ```

4. Generate a public Railway domain and verify that opening it displays the API welcome message.

### Vercel frontend

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Select the **Vite** framework preset.
4. Use `npm run build` as the build command and `dist` as the output directory.
5. Add this environment variable:

   ```env
   VITE_API_BASE_URL=https://your-railway-domain
   ```

6. Deploy, then test the metrics table and all three prediction models.
