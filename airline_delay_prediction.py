"""
=============================================================================
  PROJECT 11: AIRLINE DELAY PREDICTION USING PYSPARK MLlib
=============================================================================
  Author      : [Your Name]
  Tools       : PySpark, MLlib, Matplotlib, Seaborn, Pandas
  Dataset     : U.S. Airline On-Time Performance (BTS / Kaggle)
  Objective   : Predict flight arrival delay (in minutes) using distance,
                departure time, and weather-related features.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 │ IMPORTS & ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# PySpark core
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, DoubleType
)

# PySpark ML
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 │ INITIALIZE SPARK SESSION
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED"

spark = (
    SparkSession.builder
    .appName("AirlineDelayPrediction")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.extraJavaOptions",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED")
    .getOrCreate()
)
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 │ GENERATE SYNTHETIC DATASET  (replace with real CSV load below)
# ─────────────────────────────────────────────────────────────────────────────
#
#  HOW TO USE YOUR OWN DATASET
#  ────────────────────────────
#  Download from:  https://www.kaggle.com/datasets/usdot/flight-delays
#  Then replace the block below with:
#
#      df_raw = spark.read.csv(
#          "data/flights.csv",
#          header=True,
#          inferSchema=True
#      )
#
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2 │ Loading Dataset into Spark DataFrame")
print("="*70)

# ── Synthetic data generation ────────────────────────────────────────────────
np.random.seed(42)
N = 50_000

airlines  = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
airports  = ["ATL", "ORD", "LAX", "DFW", "JFK", "DEN", "SFO", "LAS",
             "SEA", "CLT", "PHX", "MIA", "IAH", "MCO", "EWR"]

airline_col     = np.random.choice(airlines, N)
origin_col      = np.random.choice(airports, N)
dest_col        = np.random.choice(airports, N)
distance_col    = np.random.randint(200, 3000, N).astype(float)
dep_hour_col    = np.random.randint(5, 23, N).astype(float)
dep_delay_col   = np.random.normal(10, 25, N)            # departure delay (min)
wind_speed_col  = np.random.uniform(0, 40, N)            # mph
visibility_col  = np.random.uniform(1, 10, N)            # miles
precip_col      = np.random.uniform(0, 2, N)             # inches/hr
temp_col        = np.random.uniform(10, 105, N)          # °F
month_col       = np.random.randint(1, 13, N).astype(float)
day_of_week_col = np.random.randint(1, 8, N).astype(float)

# Simulate realistic arrival delay using domain logic
arr_delay_col = (
    dep_delay_col * 0.75
    + distance_col * 0.003
    + wind_speed_col * 0.4
    - visibility_col * 1.2
    + precip_col * 5.0
    + np.random.normal(0, 10, N)
)

# Build Pandas → Spark DataFrame
pdf = pd.DataFrame({
    "airline"      : airline_col,
    "origin"       : origin_col,
    "dest"         : dest_col,
    "month"        : month_col,
    "day_of_week"  : day_of_week_col,
    "dep_hour"     : dep_hour_col,
    "distance"     : distance_col,
    "dep_delay"    : dep_delay_col.round(2),
    "wind_speed"   : wind_speed_col.round(2),
    "visibility"   : visibility_col.round(2),
    "precipitation": precip_col.round(3),
    "temperature"  : temp_col.round(1),
    "arr_delay"    : arr_delay_col.round(2),   # ← TARGET
})

# Inject realistic missing values (≈3 %)
for col in ["dep_delay", "wind_speed", "visibility", "arr_delay"]:
    idx = np.random.choice(N, int(N * 0.03), replace=False)
    pdf.loc[idx, col] = np.nan

df_raw = spark.createDataFrame(pdf)
print(f"✔  Rows loaded    : {df_raw.count():,}")
print(f"✔  Columns        : {len(df_raw.columns)}")
df_raw.printSchema()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 │ DATA CLEANING & MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3 │ Handling Missing / Incorrect Records")
print("="*70)

# ── 3a. Show null counts before cleaning ─────────────────────────────────────
print("\n  Null counts BEFORE cleaning:")
null_counts = {c: df_raw.filter(F.col(c).isNull()).count()
               for c in df_raw.columns}
for col, cnt in null_counts.items():
    if cnt > 0:
        print(f"     {col:<18}: {cnt:,} nulls")

# ── 3b. Drop rows where target (arr_delay) is null ───────────────────────────
df_clean = df_raw.filter(F.col("arr_delay").isNotNull())
print(f"\n  Rows after dropping null target : {df_clean.count():,}")

# ── 3c. Impute numeric columns with median ────────────────────────────────────
numeric_cols = ["dep_delay", "wind_speed", "visibility"]
medians = (
    df_clean.approxQuantile(numeric_cols, [0.5], 0.001)
)

for col, (median_val,) in zip(numeric_cols, medians):
    df_clean = df_clean.fillna({col: round(median_val, 2)})
    print(f"  Imputed {col:<18} with median = {median_val:.2f}")

# ── 3d. Remove impossible delay values (< -60 or > 600 mins) ─────────────────
before = df_clean.count()
df_clean = df_clean.filter(
    (F.col("arr_delay") >= -60) & (F.col("arr_delay") <= 600)
)
after = df_clean.count()
print(f"\n  Rows removed (outlier delays)   : {before - after:,}")
print(f"  Final clean dataset rows        : {after:,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 │ FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 4 │ Feature Engineering")
print("="*70)

# ── 4a. Derived features ──────────────────────────────────────────────────────
df_feat = (
    df_clean
    # Rush-hour flag (7-9 AM or 4-7 PM)
    .withColumn("is_rush_hour",
        F.when(
            (F.col("dep_hour").between(7, 9)) |
            (F.col("dep_hour").between(16, 19)),
            1.0
        ).otherwise(0.0)
    )
    # Weekend flag
    .withColumn("is_weekend",
        F.when(F.col("day_of_week").isin(6, 7), 1.0).otherwise(0.0)
    )
    # Winter months flag (Dec, Jan, Feb)
    .withColumn("is_winter",
        F.when(F.col("month").isin(12, 1, 2), 1.0).otherwise(0.0)
    )
    # Bad weather composite score
    .withColumn("weather_score",
        F.col("wind_speed") * 0.5
        + (10 - F.col("visibility")) * 1.5
        + F.col("precipitation") * 10.0
    )
)

print("  ✔  Derived features created: is_rush_hour, is_weekend, is_winter, weather_score")
df_feat.select(
    "airline", "origin", "distance", "dep_delay",
    "weather_score", "arr_delay"
).show(5, truncate=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 │ ENCODING CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5 │ Encoding Categorical Features (StringIndexer + OHE)")
print("="*70)

cat_cols = ["airline", "origin", "dest"]
indexed_cols  = [f"{c}_idx" for c in cat_cols]
encoded_cols  = [f"{c}_ohe" for c in cat_cols]

indexers = [
    StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
    for c, idx in zip(cat_cols, indexed_cols)
]
encoders = [
    OneHotEncoder(inputCol=idx, outputCol=enc, dropLast=True)
    for idx, enc in zip(indexed_cols, encoded_cols)
]

print(f"  ✔  Categorical columns encoded : {cat_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 │ ASSEMBLING FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 6 │ Assembling Feature Vector & Scaling")
print("="*70)

numeric_features = [
    "month", "day_of_week", "dep_hour", "distance",
    "dep_delay", "wind_speed", "visibility", "precipitation",
    "temperature", "is_rush_hour", "is_weekend",
    "is_winter", "weather_score"
]
all_features = numeric_features + encoded_cols

assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="raw_features",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)
print(f"  ✔  Total feature inputs        : {len(all_features)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 │ TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 7 │ Train / Test Split (80 / 20)")
print("="*70)

train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
print(f"  ✔  Training rows   : {train_df.count():,}")
print(f"  ✔  Testing rows    : {test_df.count():,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 │ BUILD ML PIPELINE & TRAIN LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 8 │ Building Pipeline & Training Linear Regression")
print("="*70)

lr = LinearRegression(
    featuresCol="features",
    labelCol="arr_delay",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.1,        # L2 regularization
    elasticNetParam=0.2  # 20% L1, 80% L2  →  ElasticNet
)

pipeline = Pipeline(stages=[
    *indexers,
    *encoders,
    assembler,
    scaler,
    lr
])

print("  Training model … (this may take a few seconds)")
model = pipeline.fit(train_df)
print("  ✔  Model trained successfully!")

# ── Coefficients summary ──────────────────────────────────────────────────────
lr_model = model.stages[-1]
print(f"\n  Intercept      : {lr_model.intercept:.4f}")
print(f"  Num features   : {len(lr_model.coefficients)}")
print(f"  Training RMSE  : {lr_model.summary.rootMeanSquaredError:.4f}")
print(f"  Training R²    : {lr_model.summary.r2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 │ MODEL EVALUATION (RMSE & R²)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 9 │ Model Evaluation on Test Set")
print("="*70)

predictions = model.transform(test_df)

evaluator_rmse = RegressionEvaluator(
    labelCol="arr_delay", predictionCol="prediction", metricName="rmse"
)
evaluator_r2   = RegressionEvaluator(
    labelCol="arr_delay", predictionCol="prediction", metricName="r2"
)
evaluator_mae  = RegressionEvaluator(
    labelCol="arr_delay", predictionCol="prediction", metricName="mae"
)

rmse = evaluator_rmse.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)

print(f"\n  {'Metric':<30} {'Value':>12}")
print(f"  {'─'*42}")
print(f"  {'Root Mean Squared Error (RMSE)':<30} {rmse:>12.4f} min")
print(f"  {'Mean Absolute Error (MAE)':<30} {mae:>12.4f} min")
print(f"  {'R² Score':<30} {r2:>12.4f}")
print(f"  {'─'*42}")

if r2 >= 0.85:
    print("\n  ✅  Excellent model performance (R² ≥ 0.85)")
elif r2 >= 0.70:
    print("\n  ✔   Good model performance (R² ≥ 0.70)")
else:
    print("\n  ⚠   Consider feature engineering or non-linear models.")

predictions.select(
    "airline", "origin", "distance", "arr_delay", "prediction"
).show(10, truncate=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 │ OPTIONAL HYPERPARAMETER TUNING (CrossValidation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 10 │ Hyperparameter Tuning (3-Fold Cross Validation)")
print("="*70)

# Rebuild pipeline without the final lr (to add grid search)
lr_cv = LinearRegression(
    featuresCol="features",
    labelCol="arr_delay",
    predictionCol="prediction",
    maxIter=50
)
pipeline_cv = Pipeline(stages=[
    *indexers,
    *encoders,
    assembler,
    scaler,
    lr_cv
])

param_grid = (
    ParamGridBuilder()
    .addGrid(lr_cv.regParam,        [0.01, 0.1, 0.5])
    .addGrid(lr_cv.elasticNetParam, [0.0,  0.5])
    .build()
)

cross_val = CrossValidator(
    estimator=pipeline_cv,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_rmse,
    numFolds=3,
    seed=42
)

print("  Running 3-fold CV over 6 parameter combinations …")
cv_model    = cross_val.fit(train_df)
best_model  = cv_model.bestModel
best_lr     = best_model.stages[-1]

print(f"\n  Best regParam         : {best_lr.getRegParam()}")
print(f"  Best elasticNetParam  : {best_lr.getElasticNetParam()}")

cv_preds = best_model.transform(test_df)
cv_rmse  = evaluator_rmse.evaluate(cv_preds)
cv_r2    = evaluator_r2.evaluate(cv_preds)
cv_mae   = evaluator_mae.evaluate(cv_preds)

print(f"\n  Post-CV RMSE : {cv_rmse:.4f}")
print(f"  Post-CV MAE  : {cv_mae:.4f}")
print(f"  Post-CV R²   : {cv_r2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 │ VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 11 │ Generating Visualizations")
print("="*70)

os.makedirs("outputs", exist_ok=True)

# Convert predictions to Pandas for plotting (sample for speed)
pdf_preds = (
    cv_preds
    .select("arr_delay", "prediction", "airline",
            "distance", "weather_score")
    .sample(fraction=0.05, seed=42)
    .toPandas()
)
pdf_preds.columns = [
    "Actual", "Predicted", "Airline", "Distance", "WeatherScore"
]
pdf_preds["Residual"] = pdf_preds["Actual"] - pdf_preds["Predicted"]

# ────────────────────────────────────────────────────
# Figure 1 │ Actual vs Predicted + Residual Analysis
# ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
fig.suptitle(
    "PROJECT 11 │ AIRLINE DELAY PREDICTION  —  Model Diagnostics",
    fontsize=16, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT = "#00c8ff"
GOLD   = "#ffd166"
RED    = "#ef476f"
GREEN  = "#06d6a0"
PANEL  = "#161b22"

# ── Plot 1: Actual vs Predicted scatter ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(PANEL)
sc = ax1.scatter(
    pdf_preds["Actual"], pdf_preds["Predicted"],
    alpha=0.5, s=15, c=pdf_preds["WeatherScore"],
    cmap="plasma", edgecolors="none"
)
lim = [
    min(pdf_preds["Actual"].min(), pdf_preds["Predicted"].min()) - 5,
    max(pdf_preds["Actual"].max(), pdf_preds["Predicted"].max()) + 5,
]
ax1.plot(lim, lim, "--", color=GREEN, lw=2, label="Perfect Prediction")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Actual Arrival Delay (min)", color="white", fontsize=11)
ax1.set_ylabel("Predicted Arrival Delay (min)", color="white", fontsize=11)
ax1.set_title("Actual vs Predicted Delay", color=ACCENT, fontsize=13, fontweight="bold")
ax1.tick_params(colors="white")
ax1.legend(fontsize=10, facecolor=PANEL, edgecolor="white", labelcolor="white")
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label("Weather Score", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Metrics text box
metrics_txt = f"RMSE  = {cv_rmse:.2f} min\nMAE   = {cv_mae:.2f} min\nR²    = {cv_r2:.4f}"
ax1.text(
    0.02, 0.97, metrics_txt, transform=ax1.transAxes,
    fontsize=10, verticalalignment="top",
    bbox=dict(facecolor=PANEL, edgecolor=ACCENT, alpha=0.9),
    color=GOLD, fontfamily="monospace"
)

# ── Plot 2: Residual distribution ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
ax2.hist(pdf_preds["Residual"], bins=40, color=ACCENT, alpha=0.75, edgecolor="none")
ax2.axvline(0, color=RED, lw=2, linestyle="--", label="Zero Error")
ax2.set_xlabel("Residual (min)", color="white", fontsize=11)
ax2.set_ylabel("Frequency", color="white", fontsize=11)
ax2.set_title("Residual Distribution", color=ACCENT, fontsize=13, fontweight="bold")
ax2.tick_params(colors="white")
ax2.legend(facecolor=PANEL, edgecolor="white", labelcolor="white")

# ── Plot 3: Delay by Airline ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
airline_avg = (
    pdf_preds.groupby("Airline")[["Actual", "Predicted"]]
    .mean()
    .sort_values("Actual", ascending=True)
)
x = range(len(airline_avg))
ax3.barh(list(x), airline_avg["Actual"],   color=ACCENT, alpha=0.7, label="Actual",    height=0.4)
ax3.barh([i + 0.4 for i in x], airline_avg["Predicted"], color=GOLD,  alpha=0.7, label="Predicted", height=0.4)
ax3.set_yticks([i + 0.2 for i in x])
ax3.set_yticklabels(airline_avg.index, color="white")
ax3.set_xlabel("Avg Delay (min)", color="white", fontsize=10)
ax3.set_title("Avg Delay by Airline", color=ACCENT, fontsize=12, fontweight="bold")
ax3.tick_params(colors="white")
ax3.legend(facecolor=PANEL, edgecolor="white", labelcolor="white", fontsize=9)

# ── Plot 4: Distance vs Delay ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
ax4.scatter(
    pdf_preds["Distance"], pdf_preds["Actual"],
    alpha=0.3, s=10, color=ACCENT, label="Actual"
)
ax4.scatter(
    pdf_preds["Distance"], pdf_preds["Predicted"],
    alpha=0.3, s=10, color=GOLD, label="Predicted"
)
ax4.set_xlabel("Distance (miles)", color="white", fontsize=10)
ax4.set_ylabel("Delay (min)", color="white", fontsize=10)
ax4.set_title("Distance vs Delay", color=ACCENT, fontsize=12, fontweight="bold")
ax4.tick_params(colors="white")
ax4.legend(facecolor=PANEL, edgecolor="white", labelcolor="white", fontsize=9)

# ── Plot 5: Residual vs Predicted (Homoscedasticity check) ────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
ax5.scatter(
    pdf_preds["Predicted"], pdf_preds["Residual"],
    alpha=0.4, s=10, color=GREEN
)
ax5.axhline(0, color=RED, lw=2, linestyle="--")
ax5.set_xlabel("Predicted Delay (min)", color="white", fontsize=10)
ax5.set_ylabel("Residual (min)", color="white", fontsize=10)
ax5.set_title("Residuals vs Fitted", color=ACCENT, fontsize=12, fontweight="bold")
ax5.tick_params(colors="white")

# Global spine styling
for ax in [ax1, ax2, ax3, ax4, ax5]:
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

plt.savefig(
    "outputs/model_diagnostics.png",
    dpi=150, bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
print("  ✔  Saved → outputs/model_diagnostics.png")

# ────────────────────────────────────────────────────
# Figure 2 │ Feature Importance via Coefficients
# ────────────────────────────────────────────────────
coef_labels = numeric_features + [f"{c}_ohe" for c in cat_cols]
coefficients = best_lr.coefficients.toArray()[:len(coef_labels)]

coef_df = pd.DataFrame({
    "Feature": coef_labels,
    "Coefficient": coefficients
}).sort_values("Coefficient", key=abs, ascending=False).head(15)

fig2, ax = plt.subplots(figsize=(12, 7), facecolor="#0d1117")
ax.set_facecolor(PANEL)
colors = [GREEN if v >= 0 else RED for v in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, alpha=0.85)
ax.axvline(0, color="white", lw=1, linestyle="--")
ax.set_xlabel("Coefficient Value", color="white", fontsize=12)
ax.set_title(
    "Top 15 Feature Coefficients  (Linear Regression)\n"
    "Green = increases delay  │  Red = reduces delay",
    color=ACCENT, fontsize=13, fontweight="bold"
)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(
    "outputs/feature_importance.png",
    dpi=150, bbox_inches="tight",
    facecolor=fig2.get_facecolor()
)
print("  ✔  Saved → outputs/feature_importance.png")
plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 │ SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 12 │ Saving Best Model to Disk")
print("="*70)

model_path = "outputs/airline_delay_model"
best_model.write().overwrite().save(model_path)
print(f"  ✔  Model saved → {model_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 │ PROJECT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 13 │ PROJECT SUMMARY")
print("="*70)
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │          AIRLINE DELAY PREDICTION — RESULTS             │
  ├─────────────────────────────────────────────────────────┤
  │  Dataset rows (clean)   : {after:>10,}                   │
  │  Features used          : {len(all_features):>10}                   │
  │  Algorithm              :  Linear Regression (MLlib)    │
  │  Regularization         :  ElasticNet (L1 + L2)         │
  │  CV Folds               :          3                    │
  ├─────────────────────────────────────────────────────────┤
  │  Test RMSE              : {cv_rmse:>10.4f} min               │
  │  Test MAE               : {cv_mae:>10.4f} min               │
  │  Test R²                : {cv_r2:>10.4f}                   │
  ├─────────────────────────────────────────────────────────┤
  │  Outputs                                                │
  │   • outputs/model_diagnostics.png                       │
  │   • outputs/feature_importance.png                      │
  │   • outputs/airline_delay_model/                        │
  └─────────────────────────────────────────────────────────┘
""")

spark.stop()
print("  Spark session stopped. Project complete ✅\n")
