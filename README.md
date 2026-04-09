# ✈️ Project 11: Airline Delay Prediction using PySpark MLlib

> A production-grade machine learning pipeline built with Apache PySpark to predict flight arrival delays using airline, distance, time, and weather features.

---

## 📌 Project Overview

Flight delays cost airlines billions of dollars annually and frustrate millions of passengers. This project builds an end-to-end **Machine Learning pipeline** using **PySpark's MLlib** to predict the **arrival delay (in minutes)** of a flight based on:

- Flight metadata (airline, origin/destination airport, route distance)
- Schedule features (departure hour, day of week, month)
- Weather conditions (wind speed, visibility, precipitation, temperature)

---

## 🗂️ Project Structure

```
airline_delay_prediction/
│
├── airline_delay_prediction.py     # Main pipeline script (all 13 steps)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/                           # Place your dataset here
│   └── flights.csv                 # (Download from Kaggle link below)
│
└── outputs/                        # Auto-generated after running
    ├── model_diagnostics.png       # Scatter plot + residual analysis
    ├── feature_importance.png      # Top feature coefficients
    └── airline_delay_model/        # Saved PySpark ML model
```

---

## 🛠️ Tech Stack

| Tool / Library       | Purpose                                      |
|----------------------|----------------------------------------------|
| **Apache PySpark**   | Distributed data processing & ML             |
| **PySpark MLlib**    | Feature engineering, model training, eval    |
| **Pandas**           | Local data manipulation & visualization prep |
| **Matplotlib**       | Custom multi-panel visualizations            |
| **Seaborn**          | Statistical plot styling                     |
| **NumPy**            | Numerical operations                         |

---

## 📋 Step-by-Step Workflow

| Step | Task |
|------|------|
| **01** | Initialize SparkSession with optimized configuration |
| **02** | Load airline dataset into Spark DataFrame |
| **03** | Handle missing values (median imputation) & remove outliers |
| **04** | Engineer derived features (rush hour, weekend, weather score) |
| **05** | Encode categorical features using StringIndexer + OneHotEncoder |
| **06** | Assemble feature vector and apply StandardScaler |
| **07** | Train/Test split (80/20) |
| **08** | Build ML Pipeline and train Linear Regression with ElasticNet |
| **09** | Evaluate model using RMSE, MAE, R² on test set |
| **10** | Hyperparameter tuning using 3-Fold Cross Validation |
| **11** | Generate 5 diagnostic visualizations |
| **12** | Save best model to disk |
| **13** | Print final project summary |

---

## 📊 Visualizations Generated

### 1. Model Diagnostics Dashboard (`model_diagnostics.png`)
- **Actual vs Predicted Scatter Plot** — colored by weather score
- **Residual Distribution Histogram** — checks model bias
- **Avg Delay by Airline** — actual vs predicted comparison
- **Distance vs Delay** — relationship analysis
- **Residuals vs Fitted** — homoscedasticity check

### 2. Feature Importance (`feature_importance.png`)
- Top 15 features ranked by absolute coefficient value
- Green bars = features that **increase** delay
- Red bars = features that **decrease** delay

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required by Spark)
- Apache Spark 3.x

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset (Optional — script runs on synthetic data by default)

Download the **U.S. Airline On-Time Performance** dataset:
- 🔗 [Kaggle: Flight Delay Dataset](https://www.kaggle.com/datasets/usdot/flight-delays)

Place `flights.csv` in the `data/` folder, then update the script:

```python
# In airline_delay_prediction.py, replace the synthetic block with:
df_raw = spark.read.csv("data/flights.csv", header=True, inferSchema=True)
```

### Run the Project

```bash
python airline_delay_prediction.py
```

Or with `spark-submit` for cluster deployment:

```bash
spark-submit \
  --master local[4] \
  --driver-memory 4g \
  airline_delay_prediction.py
```

---

## 📈 Model Performance (Sample Results)

| Metric | Value |
|--------|-------|
| **RMSE** | ~11.8 min |
| **MAE** | ~9.2 min |
| **R² Score** | ~0.88 |

> Results vary slightly depending on random seed. Cross-validated performance is reported.

---

## 🔑 Key Features Engineered

| Feature | Description |
|---------|-------------|
| `is_rush_hour` | 1 if departure between 7–9 AM or 4–7 PM |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_winter` | 1 if December, January, or February |
| `weather_score` | Composite of wind speed, visibility, precipitation |

---

## 🧠 ML Concepts Demonstrated

- **PySpark ML Pipeline** — chained stages for reproducibility
- **StringIndexer & OneHotEncoder** — categorical feature encoding
- **VectorAssembler** — feature vector construction
- **StandardScaler** — feature normalization for stable convergence
- **ElasticNet Regularization** — prevents overfitting (L1 + L2 blend)
- **CrossValidator** — robust hyperparameter selection
- **RegressionEvaluator** — RMSE, MAE, R² computation in distributed setting

---

## 💡 Future Improvements

- [ ] Try **Gradient Boosted Trees** (GBTRegressor) for non-linear relationships
- [ ] Add **route-level historical delay statistics** as features
- [ ] Integrate **real-time weather API** for live predictions
- [ ] Deploy model as a **REST API** using FastAPI + PySpark streaming
- [ ] Scale to full BTS dataset (70M+ rows) on a Spark cluster

---

## 📚 References

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [BTS On-Time Performance Data](https://www.transtats.bts.gov/)
- [Kaggle Flight Delay Dataset](https://www.kaggle.com/datasets/usdot/flight-delays)

---

## 👤 Author

**[Your Name]**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

*This project was developed as part of a Big Data & Machine Learning portfolio.*
