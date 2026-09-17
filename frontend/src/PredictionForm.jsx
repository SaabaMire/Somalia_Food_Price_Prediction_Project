import React, { useState, useEffect } from "react";

// === API Configuration ===
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

// === Initial Data ===
const INITIAL_FORM_DATA = {
  commodity: "",
  market: "",
  unit: "",
  year: "",
  month: "",
  model: "gbr",
};

// === Metrics Table ===
const MetricsTable = ({ metrics }) => {
  if (!metrics) return null;

  let bestModel = "";
  let maxR2 = -1;
  for (const [model, data] of Object.entries(metrics)) {
    if (data.R2 > maxR2) {
      maxR2 = data.R2;
      bestModel = model;
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-xl border border-gray-100 w-full max-w-4xl ">
      <h2 className="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">
        Model Performance Overview
      </h2>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-teal-50">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-600 uppercase">
              Model
            </th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-600 uppercase">
              R² Score
            </th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-600 uppercase">
              MAE 
            </th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-600 uppercase">
              RMSE
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {Object.entries(metrics).map(([model, data]) => (
            <tr
              key={model}
              className={model === bestModel ? "bg-orange-50 font-medium" : ""}
            >
              <td
                className={`px-3 py-2 whitespace-nowrap text-sm capitalize ${
                  model === bestModel ? "text-teal-700" : "text-gray-900"
                }`}
              >
                {model.replace("_", " ")}
              </td>
              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700">
                {(data.R2 * 100).toFixed(1)}%
              </td>
              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700">
                {data.MAE.toLocaleString("en-US", {
                  maximumFractionDigits: 0,
                })}{" "}
                SLS
              </td>
              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700">
                {data.RMSE.toLocaleString("en-US", {
                  maximumFractionDigits: 0,
                })}{" "}
                SLS
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-4 text-xs text-gray-500">
        *RF (Random Forest Regressor) is the most accurate model based on
        test data.
      </p>
    </div>
  );
};

// === Main Component ===
const PredictionForm = () => {
  const [formData, setFormData] = useState(INITIAL_FORM_DATA);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch Metrics from Backend
  useEffect(() => {
    fetch(`${API_BASE_URL}/metrics`)
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error("Could not fetch metrics:", err));
  }, []);

  // Handle Input Changes (limit year/month to digits)
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]:
        name === "year" || name === "month" ? value.replace(/\D/g, "") : value,
    }));
  };

  // Handle Form Submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    const { model, ...payload } = formData;

    // 1️⃣ Required fields validation
    const requiredFields = ["market", "commodity", "unit", "year", "month"];
    const missing = requiredFields.filter((f) => !payload[f]?.trim());
    if (missing.length > 0) {
      setError(`Please fill all required fields: ${missing.join(", ")}.`);
      setLoading(false);
      return;
    }

    // 2️⃣ Year & Month validation
    const year = Number(payload.year);
    const month = Number(payload.month);

    if (!year || year < 1990 || year > 2100) {
      setError("Year must be a valid number between 1990 and 2100.");
      setLoading(false);
      return;
    }

    if (!month || month < 1 || month > 12) {
      setError("Month must be between 1 (January) and 12 (December).");
      setLoading(false);
      return;
    }

    // 3️⃣ API request
    try {
      const response = await fetch(`${API_BASE_URL}/predict?model=${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data.predicted_price);
      } else {
        setError(data.error || "An unknown error occurred during prediction.");
      }
    } catch (err) {
      setError(
        "Network error: Could not connect to the API. Please check if Flask server is running."
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Render Input
  const renderInput = (name, label, placeholder, type = "text") => (
    <div className="flex flex-col">
      <label
        htmlFor={name}
        className="text-sm font-medium text-gray-600 mb-1"
      >
        {label}
      </label>
      <input
        type={type}
        name={name}
        value={formData[name]}
        onChange={handleChange}
        placeholder={placeholder}
        min={name === "month" ? 1 : name === "year" ? 1990 : undefined}
        max={name === "month" ? 12 : name === "year" ? 2100 : undefined}
        className="mt-1 block w-full pl-3 pr-4 py-2 text-base border-gray-300 focus:outline-none focus:ring-teal-500 focus:border-teal-500 sm:text-sm rounded-md shadow-sm transition duration-150"
        required
      />
    </div>
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* === LEFT: FORM === */}
      <div className="lg:col-span-2">
        <form
          onSubmit={handleSubmit}
          className="bg-white p-8 rounded-xl shadow-2xl border border-teal-200 h-full"
        >
          <h2 className="text-3xl font-extrabold text-teal-800 mb-8 border-b pb-3">
            Predict Food Price 
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {renderInput("market", "1. Market Name", "Enter Market Name")}
            {renderInput("commodity", "2. Commodity Name", "Enter Commodity Name")}
            {renderInput("unit", "3. Measurement Unit", "Enter Unit (e.g., KG)")}
            {renderInput("year", "4. Year", "Enter Year (e.g., 2025)", "number")}
            {renderInput("month", "5. Month", "Enter Month (1–12)", "number")}
          </div>

          {/* === MODEL SELECTION === */}
          <div className="flex flex-col mb-8 p-4 bg-teal-50 rounded-lg border border-teal-200">
            <label className="text-lg font-bold text-teal-700 mb-2">
              Select ML Model
            </label>
            <div className="flex space-x-6">
              <ModelRadio
                value="gbr"
                label="Gradient Boosting (GBR)"
                checked={formData.model === "gbr"}
                onChange={handleChange}
              />
              <ModelRadio
                value="rf"
                label="Random Forest (RF)"
                checked={formData.model === "rf"}
                onChange={handleChange}
              />
              <ModelRadio
                value="lr"
                label="Linear Regression (LR)"
                checked={formData.model === "lr"}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* === SUBMIT BUTTON === */}
          <button
            type="submit"
            disabled={loading}
            className={`w-full py-4 px-4 rounded-xl text-white font-bold text-lg tracking-wider transition duration-300 ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-teal-600 hover:bg-teal-700 shadow-lg hover:shadow-xl"
            }`}
          >
            {loading
              ? "CALCULATING PRICE..."
              : "PREDICT PRICE IN SOMALI SHILLINGS (SLS)"}
          </button>
        </form>
      </div>

      {/* === RIGHT: RESULT & METRICS === */}
      <div className="lg:col-span-1 space-y-8">
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md shadow-md">
            <p className="font-bold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {prediction !== null && (
          <div className="bg-orange-500 p-8 rounded-xl text-white shadow-2xl text-center border-b-8 border-orange-700">
            <p className="text-md uppercase tracking-widest font-bold">
              Predicted Price
            </p>
            <p className="text-6xl font-extrabold mt-2 leading-tight">
              {prediction.toLocaleString("en-US", {
                maximumFractionDigits: 0,
              })}
            </p>
            <p className="text-lg font-medium mt-1">SLS</p>
            <p className="text-sm mt-3 font-medium opacity-80">
              Model Used: {formData.model.toUpperCase()}
            </p>
          </div>
        )}

        <MetricsTable metrics={metrics} />
      </div>
    </div>
  );
};

// === Model Radio Button ===
const ModelRadio = ({ value, label, checked, onChange }) => (
  <label className="inline-flex items-center cursor-pointer">
    <input
      type="radio"
      name="model"
      value={value}
      checked={checked}
      onChange={onChange}
      className="form-radio h-4 w-4 text-teal-600 focus:ring-teal-500 transition duration-150 ease-in-out"
    />
    <span className="ml-2 text-sm text-gray-700 font-medium">{label}</span>
  </label>
);

export default PredictionForm;
