***
# Somalia Food Price Prediction Project: Model Development and Deployment

**Author:** [Sabirin Mire Abukar]  
**Date:** October 9, 2025  
**Project Title:** Time Series Price Prediction using Ensemble Methods for Food Security Planning

---

## 1. Introduction: Problem Statement and Motivation

### 1.1 Problem Statement
The primary challenge addressed by this project is the **volatility and unpredictability of essential food item prices** within the dynamic markets of Somalia. Price stability is crucial for food security, humanitarian aid planning, and economic stability. By building machine learning models capable of accurately predicting retail prices weeks or months in advance, we aim to provide actionable insights for stakeholders.

### 1.2 Motivation
Predictive modeling in this domain offers significant value:
1.  **Humanitarian Aid:** Non-governmental organizations (NGOs) can better allocate resources, plan budgets, and optimize supply chain logistics by anticipating future price surges.
2.  **Economic Planning:** Local governments and businesses gain a tool to mitigate risk associated with commodity price fluctuations, potentially stabilizing local food supplies.
3.  **Risk Mitigation:** Providing advance warning on price trends allows farmers, traders, and consumers to make informed decisions, reducing the economic impact of market shocks.

This project is deployed as a working API with a simple frontend, making the predictive intelligence accessible and immediately useful.

## 2. Dataset & Preprocessing

### 2.1 Dataset Source and Size
The data utilized is the **World Food Programme (WFP) Food Prices data for Somalia**, accessed via Kaggle or the WFP data repository.

* **Source:** World Food Programme (WFP) Price Monitoring Tool.
* **Size:** The raw dataset contains over **11,000 samples** of monthly price observations spanning several decades, ensuring statistical robustness.

### 2.2 Features
The final prediction relies on the following key features:
* **Categorical Features:** `market`, `admin1` (Region), `admin2` (District), `category`, `commodity`, `unit`, `pricetype`.
* **Numerical Features:** `latitude`, `longitude`, `year`, and `month`.
* **Target Variable:** `price` (in Somali Shillings, SLS).

### 2.3 Preprocessing Steps

The data preprocessing was critical for preparing the highly categorical and geographically specific dataset for machine learning. The key steps executed in `preprocessing.py` are:

1.  **Data Cleaning:** Dropped irrelevant columns (e.g., `usdprice`, `currency`, `priceflag`).
2.  **Time Feature Extraction:** Extracted dedicated **`year`** and **`month`** columns from the date.
3.  **Handling Missing Values:** Missing numerical values (like `latitude`/`longitude`) were imputed using the column **median**. Missing categorical values were filled with a placeholder string.
4.  **Outlier Management:** Outliers in the target variable, **`price`**, were capped using the Interquartile Range (IQR) method to prevent extreme values from distorting training.
5.  **One-Hot Encoding (OHE):** All categorical features were transformed into numerical format using OHE.
6.  **Feature Scaling:** All resulting numerical input columns were scaled using the **StandardScaler**.
7.  **Training Column Preservation:** The final list of feature columns (`TRAIN_COLUMNS`) was saved to a JSON file to guarantee structure parity during API inference.

## 3. Algorithms and Why They Were Chosen

Three distinct regression models were trained and compared to establish the optimal approach for this non-linear time series problem.

### 3.1 Linear Regression (LR)
* **Type:** Simple, parametric linear model.
* **Rationale:** Chosen as a **baseline model** to determine the minimal predictive power possible. It provides the quickest training time and simplest interpretation.

### 3.2 Gradient Boosting Regressor (GBR)
* **Type:** Advanced ensemble tree-based model.
* **Rationale:** Chosen for its **high-performance potential**. GBR builds trees sequentially, with each new tree attempting to correct the residual errors of the previous ensemble. It often achieves state-of-the-art results in structured data prediction tasks.

### 3.3 Random Forest Regressor (RF)
* **Type:** Ensemble, non-parametric tree-based model.
* **Rationale:** Chosen for its **robustness and parallelizability**. RF builds multiple independent decision trees and averages their predictions. It is highly effective at capturing complex, non-linear feature interactions without being as prone to overfitting as GBR.

## 4. Results and Discussion

### 4.1 Performance Metrics

The models were rigorously evaluated on a dedicated 20% test set using $R^2$ (Coefficient of Determination) and MAE (Mean Absolute Error).

| Model | $R^2$ Score (Closer to 1.0 is Better) | MAE (Mean Absolute Error, in SLS) | Performance Assessment |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | $98.1\%$ | $1,049$ SLS | **Best Performer** (Near-perfect fit) |
| **Gradient Boosting (GBR)** | $89.5\%$ | $3,519$ SLS | Strong Performer (Excellent, but slightly worse than RF) |
| **Linear Regression (LR)** | $80.7\%$ | $5,194$ SLS | Good Baseline (Struggles with non-linearity) |

### 4.2 Discussion: Which Performed Better?
The **Random Forest Regressor (RF)** is the clear superior model, achieving an $R^2$ of $98.1\%$ and an average prediction error (MAE) of just 1,049 SLS.

* The performance difference between the ensemble methods (**RF** and **GBR**) and the simple linear model (**LR**) highlights that the price prediction problem is **highly non-linear**. Factors like market location and commodity type interact in complex ways that only tree-based models can effectively capture.
* While GBR is often the best model, **RF slightly outperformed GBR** in this specific dataset. This suggests that the aggregation and regularization inherent in the Random Forest approach provided better generalization and robustness against any potential noise or outliers remaining in the price data. RF's lower MAE makes it the recommended choice for production deployment.

### 4.3 Sanity Checks (Sample Predictions)

To validate the model's practical output, three arbitrary samples from the test set were compared across all three models:

| Data Row | Actual Price (SLS) | LR Prediction (SLS) | RF Prediction (SLS) | GBR Prediction (SLS) |
| :--- | :--- | :--- | :--- | :--- |
| **Row 5** | $36,000$ | $30,870$ | $35,108$ | $33,185$ |
| **Row 100** | $7,400$ | $8,186$ | $7,402$ | $7,731$ |
| **Row 200** | $26,500$ | $27,830$ | $26,057$ | $31,665$ |

**Conclusion:** The sanity checks confirm the metrics. The **RF model** consistently provides the closest predictions (e.g., $7,402$ vs. $7,400$ actual). The **LR model** shows significant errors (e.g., $30,870$ vs. $36,000$ actual). The GBR model performed well but had a larger miss on Row 200 ($31,665$), further validating the choice of Random Forest for deployment due to its lower average error and higher $R^2$.

## 5. Deployment Notes

The selected models were exposed via a lightweight **Flask API** to ensure fast, accessible inference.

### 5.1 API Structure
* **Framework:** Flask
* **Model Access:** Models are loaded using `joblib` once at API startup.
* **Endpoint:** The primary inference route is `POST /predict?model={model_name}`, allowing users to dynamically switch between the trained algorithms (LR, RF, and GBR).

### 5.2 The `/predict` Endpoint Logic
* **Input:** Accepts a JSON payload containing only the 5 simplified features from the user.
* **Core Function:** The request data is processed by the critical `prepare_features_from_raw` function (`utils.py`). This function handles:
    1.  **Default Lookup:** Injecting the 6 hidden features (like `latitude`, `admin1`) based on the provided `market` and `commodity` using smart defaults.
    2.  **Structural Integrity:** Reconstructing the exact 100+ column input array required by the model (including One-Hot Encoded columns and scaled numerical data), guaranteeing the input shape always matches the training data structure.

### 5.3 Frontend Integration
A simple, functional React application was implemented using Tailwind CSS. This frontend component:
1.  Provides the user interface for input and model selection.
2.  Includes **client-side validation** for the `year` and `month` fields to prevent invalid inputs (e.g., negative numbers, months outside 1-12) from hitting the API.
3.  Fetches the performance metrics for display.
4.  Calls the `/predict` API endpoint to render the selected model's forecast in real-time.

## 6. Lessons Learned

### 6.1 Challenges Faced and Improvements
The greatest technical challenge lay in maintaining consistency across the development lifecycle, specifically regarding **categorical feature handling**.

* **Challenge:** When performing One-Hot Encoding during training, a fixed set of columns is generated (e.g., `market_Bakaara`, `market_Hargeysa`). If the live API receives a request for a new, unknown market, the input array size would change, causing the model to crash.
* **Improvement:** The solution, implemented in `utils.py`, was to use a pre-saved list of all trained column names (`TRAIN_COLUMNS`). The API input is initialized as an array of zeros with the correct length, and only the columns matching the input features are set to one. This guarantees that the API input shape always matches the model's expected input shape, ensuring robustness.

### 6.2 Key Takeaways
1.  **Ensemble Methods are Superior for Volatility:** For complex, high-variability regression tasks like commodity price forecasting, ensemble techniques (RF and GBR) drastically outperform simple models like Linear Regression.
2.  **Deployment Requires Rigor:** Successful deployment is entirely dependent on ensuring the preprocessing steps (scaling, encoding, and ordering) executed in the API are an **exact, immutable clone** of the steps performed during training.
3.  **Validation for Time Series:** Integrating client-side checks for time-based features (like month and year) is essential for a predictive model, as users often test boundaries that might lead to API errors. This small frontend detail dramatically improves the user experience and API stability.

This project successfully transitioned a complex data science problem into a functional, validated, and deployable machine learning service ready for real-world application.