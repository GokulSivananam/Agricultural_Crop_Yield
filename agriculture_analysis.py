# =============================================================================
# AGRICULTURE CROP YIELD ANALYSIS
# Why is Crop Yield Different?
# =============================================================================
# This script performs a complete end-to-end data science analysis on
# agricultural data to understand factors that influence crop yield.
# =============================================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ----------------------------
# GLOBAL PATHS
# ----------------------------
DATA_PATH = os.path.join("data", "agriculture.csv")
VIZ_DIR = os.path.join("outputs", "visualizations")
MODEL_DIR = os.path.join("outputs", "model")

# Ensure output directories exist
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Expected columns
REQUIRED_COLUMNS = [
    "Farm_ID", "Region", "Crop_Type", "Rainfall", "Soil_Properties",
    "Irrigation", "Fertiliser_Use", "Temperature", "Crop_Yield"
]

# Numerical feature columns (excluding Farm_ID and target)
NUMERICAL_COLS = ["Rainfall", "Soil_Properties", "Irrigation", "Fertiliser_Use", "Temperature", "Crop_Yield"]
FEATURE_COLS = ["Region", "Crop_Type", "Rainfall", "Soil_Properties", "Irrigation", "Fertiliser_Use", "Temperature"]
TARGET_COL = "Crop_Yield"


# =============================================================================
# HELPER: Section header printer
# =============================================================================
def print_section(number, title):
    print(f"\n{'='*50}")
    print(f"  {number}. {title}")
    print(f"{'='*50}")


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_data():
    """Load the agriculture CSV and validate required columns."""
    print_section(1, "Dataset Information")

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATA_PATH}")
        print("Please place agriculture.csv inside the 'data/' folder and re-run.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Strip whitespace from column names immediately after loading
    df.columns = df.columns.str.strip()

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] The following required columns are missing from the dataset:")
        for col in missing_cols:
            print(f"  - {col}")
        sys.exit(1)

    print(f"\nDataset loaded successfully from: {DATA_PATH}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn Names  : {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nDescriptive Statistics:")
    print(df.describe(include="all"))

    return df


# =============================================================================
# 2. DATA CLEANING
# =============================================================================
def clean_data(df):
    """Clean and validate the dataset."""
    print_section(2, "Data Cleaning Summary")

    original_shape = df.shape

    # --- Strip column name spaces (already done in load, kept for safety) ---
    df.columns = df.columns.str.strip()

    # --- Missing values ---
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values per column:")
    print(missing)
    print(f"Total missing values: {total_missing}")

    # Fill missing numerical values with column median
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  -> Filled {col} missing values with median ({median_val:.2f})")

    # Fill missing categorical values with mode
    for col in ["Region", "Crop_Type"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  -> Filled {col} missing values with mode ('{mode_val}')")

    # --- Duplicate rows ---
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"  -> {duplicate_count} duplicate rows removed.")

    # --- Invalid / negative values check ---
    print(f"\nChecking for invalid (negative) values in numerical columns:")
    for col in ["Rainfall", "Soil_Properties", "Irrigation", "Fertiliser_Use", "Temperature", "Crop_Yield"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"  [WARNING] {col}: {neg_count} negative values found.")
            else:
                print(f"  {col}: No negative values. OK.")

    # --- Strip string whitespace in categorical columns ---
    for col in ["Region", "Crop_Type"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)

    print(f"\n--- Cleaning Summary ---")
    print(f"  Original shape : {original_shape[0]} rows x {original_shape[1]} columns")
    print(f"  Cleaned shape  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Rows removed   : {original_shape[0] - df.shape[0]}")
    print(f"  Missing values remaining: {df.isnull().sum().sum()}")

    return df


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
def perform_eda(df):
    """Print EDA summaries for key variables."""
    print_section(3, "Descriptive Statistics (after cleaning)")
    print(df[NUMERICAL_COLS].describe())

    print_section(4, "Regional Analysis")
    region_yield = df.groupby("Region")[TARGET_COL].mean().sort_values(ascending=False)
    print("\nAverage Crop Yield by Region:")
    print(region_yield.to_string())
    best_region = region_yield.idxmax()
    worst_region = region_yield.idxmin()
    print(f"\n  Highest average-yield region : {best_region} ({region_yield[best_region]:.2f})")
    print(f"  Lowest  average-yield region : {worst_region} ({region_yield[worst_region]:.2f})")

    print_section(5, "Crop Analysis")
    crop_yield = df.groupby("Crop_Type")[TARGET_COL].mean().sort_values(ascending=False)
    print("\nAverage Crop Yield by Crop Type:")
    print(crop_yield.to_string())
    best_crop = crop_yield.idxmax()
    worst_crop = crop_yield.idxmin()
    print(f"\n  Highest average-yield crop : {best_crop} ({crop_yield[best_crop]:.2f})")
    print(f"  Lowest  average-yield crop : {worst_crop} ({crop_yield[worst_crop]:.2f})")

    print_section(8, "Correlation Analysis")
    corr_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    print("\nCorrelation with Crop_Yield:")
    corr_with_yield = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print(corr_with_yield.round(3))

    return region_yield, crop_yield, corr_matrix


# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================
def create_visualizations(df, corr_matrix):
    """Create and save all required visualizations."""

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (9, 5)})

    # ------------------------------------------------------------------
    # Viz 1: Average Crop Yield by Region - Bar Chart
    # ------------------------------------------------------------------
    region_yield = df.groupby("Region")[TARGET_COL].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.bar(region_yield.index, region_yield.values,
                  color=sns.color_palette("Blues_d", len(region_yield)))
    ax.set_title("Average Crop Yield by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Average Crop Yield", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path1 = os.path.join(VIZ_DIR, "yield_by_region.png")
    plt.savefig(path1)
    plt.close()
    print(f"  Saved: {path1}")

    # ------------------------------------------------------------------
    # Viz 2: Crop Yield by Crop Type - Box Plot
    # ------------------------------------------------------------------
    crop_order = (df.groupby("Crop_Type")[TARGET_COL].median()
                  .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Crop_Type", y=TARGET_COL, order=crop_order,
                palette="Set2", ax=ax)
    ax.set_title("Crop Yield Distribution by Crop Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop Type", fontsize=11)
    ax.set_ylabel("Crop Yield", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path2 = os.path.join(VIZ_DIR, "yield_by_crop.png")
    plt.savefig(path2)
    plt.close()
    print(f"  Saved: {path2}")

    # ------------------------------------------------------------------
    # Viz 3: Fertiliser Use vs Crop Yield - Scatter Plot (hue = Crop_Type)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    crop_types = df["Crop_Type"].unique()
    palette = sns.color_palette("tab10", len(crop_types))
    for i, crop in enumerate(crop_types):
        subset = df[df["Crop_Type"] == crop]
        ax.scatter(subset["Fertiliser_Use"], subset[TARGET_COL],
                   label=crop, alpha=0.6, s=30, color=palette[i])
    ax.set_title("Fertiliser Use vs Crop Yield (by Crop Type)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fertiliser Use", fontsize=11)
    ax.set_ylabel("Crop Yield", fontsize=11)
    ax.legend(title="Crop Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path3 = os.path.join(VIZ_DIR, "fertiliser_vs_yield.png")
    plt.savefig(path3)
    plt.close()
    print(f"  Saved: {path3}")

    # ------------------------------------------------------------------
    # Viz 4: Irrigation vs Crop Yield - Scatter Plot (hue = Region)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    regions = df["Region"].unique()
    palette2 = sns.color_palette("Set1", len(regions))
    for i, region in enumerate(regions):
        subset = df[df["Region"] == region]
        ax.scatter(subset["Irrigation"], subset[TARGET_COL],
                   label=region, alpha=0.6, s=30, color=palette2[i])
    ax.set_title("Irrigation vs Crop Yield (by Region)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Irrigation", fontsize=11)
    ax.set_ylabel("Crop Yield", fontsize=11)
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path4 = os.path.join(VIZ_DIR, "irrigation_vs_yield.png")
    plt.savefig(path4)
    plt.close()
    print(f"  Saved: {path4}")

    # ------------------------------------------------------------------
    # Viz 5: Correlation Heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap of Numerical Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path5 = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    plt.savefig(path5)
    plt.close()
    print(f"  Saved: {path5}")


# =============================================================================
# 6. FERTILISER ANALYSIS
# =============================================================================
def analyze_fertilizer(df):
    """Analyze the relationship between fertilizer use and crop yield."""
    print_section(6, "Fertilizer Analysis")

    corr_val = df["Fertiliser_Use"].corr(df[TARGET_COL])
    print(f"\nCorrelation between Fertiliser_Use and Crop_Yield: {corr_val:.4f}")

    # Bin fertilizer use into low / medium / high based on tertiles
    df["Fertiliser_Level"] = pd.qcut(df["Fertiliser_Use"], q=3,
                                     labels=["Low", "Medium", "High"])
    avg_yield_by_fert = df.groupby("Fertiliser_Level", observed=True)[TARGET_COL].mean()
    print("\nAverage Crop Yield by Fertiliser Level (tertile bins):")
    print(avg_yield_by_fert.round(2))

    # Identify high fertilizer + low yield farms
    fert_high_threshold = df["Fertiliser_Use"].quantile(0.75)
    yield_low_threshold = df[TARGET_COL].quantile(0.25)
    inefficient_fert = df[
        (df["Fertiliser_Use"] >= fert_high_threshold) &
        (df[TARGET_COL] <= yield_low_threshold)
    ][["Farm_ID", "Region", "Crop_Type", "Fertiliser_Use", TARGET_COL]]

    print(f"\nFarms with High Fertiliser Use (>= {fert_high_threshold:.2f}) "
          f"but Low Crop Yield (<= {yield_low_threshold:.2f}):")
    if inefficient_fert.empty:
        print("  No such farms found in this dataset.")
    else:
        print(inefficient_fert.to_string(index=False))

    print("\n  Note: Correlation does not prove causation. "
          "Other factors (soil, rainfall, temperature) also influence yield.")

    # Drop temporary column
    df.drop(columns=["Fertiliser_Level"], inplace=True)

    return corr_val


# =============================================================================
# 7. IRRIGATION ANALYSIS
# =============================================================================
def analyze_irrigation(df):
    """Investigate whether more irrigation always improves crop yield."""
    print_section(7, "Irrigation Analysis")

    corr_val = df["Irrigation"].corr(df[TARGET_COL])
    print(f"\nCorrelation between Irrigation and Crop_Yield: {corr_val:.4f}")

    # Bin irrigation into low / medium / high based on tertiles
    df["Irrigation_Level"] = pd.qcut(df["Irrigation"], q=3,
                                     labels=["Low", "Medium", "High"])
    avg_yield_by_irr = df.groupby("Irrigation_Level", observed=True)[TARGET_COL].mean()
    print("\nAverage Crop Yield by Irrigation Level (tertile bins):")
    print(avg_yield_by_irr.round(2))

    # Identify high irrigation + low yield farms
    irr_high_threshold = df["Irrigation"].quantile(0.75)
    yield_low_threshold = df[TARGET_COL].quantile(0.25)
    inefficient_irr = df[
        (df["Irrigation"] >= irr_high_threshold) &
        (df[TARGET_COL] <= yield_low_threshold)
    ][["Farm_ID", "Region", "Crop_Type", "Irrigation", TARGET_COL]]

    print(f"\nFarms with High Irrigation (>= {irr_high_threshold:.2f}) "
          f"but Low Crop Yield (<= {yield_low_threshold:.2f}):")
    if inefficient_irr.empty:
        print("  No such farms found in this dataset.")
    else:
        print(inefficient_irr.to_string(index=False))

    print("\n  Note: Correlation does not prove causation. "
          "Additional irrigation does not always improve yield — "
          "over-irrigation or poor soil drainage can reduce it.")

    # Drop temporary column
    df.drop(columns=["Irrigation_Level"], inplace=True)

    return corr_val


# =============================================================================
# 8. INEFFICIENT FARMING PRACTICES
# =============================================================================
def identify_inefficient_farms(df):
    """Flag farms that use high inputs but achieve below-average yield."""
    print_section(9, "Inefficient Farming Practices")

    fert_threshold = df["Fertiliser_Use"].quantile(0.75)
    irr_threshold = df["Irrigation"].quantile(0.75)
    yield_threshold = df[TARGET_COL].quantile(0.25)

    print(f"\nThresholds used (percentile-based):")
    print(f"  High Fertiliser_Use  : >= {fert_threshold:.2f}  (75th percentile)")
    print(f"  High Irrigation      : >= {irr_threshold:.2f}  (75th percentile)")
    print(f"  Low Crop_Yield       : <= {yield_threshold:.2f}  (25th percentile)")

    # Rule 1: High fertilizer + low yield
    rule1 = df[
        (df["Fertiliser_Use"] >= fert_threshold) &
        (df[TARGET_COL] <= yield_threshold)
    ][["Farm_ID", "Region", "Crop_Type", "Fertiliser_Use", "Irrigation", TARGET_COL]]

    # Rule 2: High irrigation + low yield
    rule2 = df[
        (df["Irrigation"] >= irr_threshold) &
        (df[TARGET_COL] <= yield_threshold)
    ][["Farm_ID", "Region", "Crop_Type", "Fertiliser_Use", "Irrigation", TARGET_COL]]

    # Combine and deduplicate
    inefficient = pd.concat([rule1, rule2]).drop_duplicates(subset="Farm_ID")

    print(f"\nPotentially Inefficient Farms identified: {len(inefficient)}")
    if inefficient.empty:
        print("  No inefficient farms identified based on these thresholds.")
    else:
        print(inefficient.to_string(index=False))


# =============================================================================
# 9. RANDOM FOREST REGRESSION
# =============================================================================
def train_random_forest(df):
    """Encode features, train a Random Forest model, and return model + test data."""
    print_section(10, "Random Forest Model")

    # Copy to avoid modifying original
    model_df = df[FEATURE_COLS + [TARGET_COL]].copy()

    # Label-encode categorical columns
    le_region = LabelEncoder()
    le_crop = LabelEncoder()
    model_df["Region"] = le_region.fit_transform(model_df["Region"])
    model_df["Crop_Type"] = le_crop.fit_transform(model_df["Crop_Type"])

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining set size : {X_train.shape[0]} rows")
    print(f"Testing  set size : {X_test.shape[0]} rows")
    print(f"Features used     : {FEATURE_COLS}")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("\nRandom Forest model trained successfully.")

    return rf_model, X_test, y_test, FEATURE_COLS


# =============================================================================
# 10. MODEL EVALUATION
# =============================================================================
def evaluate_model(rf_model, X_test, y_test):
    """Evaluate model and save Actual vs Predicted scatter plot."""
    print_section(11, "Model Evaluation")

    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"  R2 Score                  : {r2:.4f}")

    if r2 >= 0.75:
        print("  Model performance: Good (R2 >= 0.75)")
    elif r2 >= 0.50:
        print("  Model performance: Moderate (R2 0.50-0.75)")
    else:
        print("  Model performance: Low (R2 < 0.50) - may need more features or tuning")

    # Actual vs Predicted scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue", s=30, label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Crop Yield", fontsize=11)
    ax.set_ylabel("Predicted Crop Yield", fontsize=11)
    ax.set_title("Actual vs Predicted Crop Yield", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    return mae, rmse, r2, y_pred


# =============================================================================
# 11. FEATURE IMPORTANCE
# =============================================================================
def analyze_feature_importance(rf_model, feature_names):
    """Extract, display, and plot feature importances."""
    print_section(12, "Feature Importance")

    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nFeature Importances (sorted by importance):")
    print(importance_df.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(importance_df["Feature"][::-1],
                   importance_df["Importance"][::-1],
                   color=sns.color_palette("viridis", len(importance_df)))
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Random Forest - Feature Importance", fontsize=14, fontweight="bold")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    return importance_df


# =============================================================================
# 12. KEY INSIGHTS
# =============================================================================
def generate_insights(df, region_yield, crop_yield, corr_matrix, importance_df,
                       fert_corr, irr_corr, r2):
    """Print data-driven insights based on analysis results."""
    print_section(13, "Key Insights")

    corr_yield = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(
        ascending=False, key=abs
    )
    top_corr_feature = corr_yield.index[0]
    top_corr_value = corr_yield.iloc[0]

    top_region = region_yield.idxmax()
    bot_region = region_yield.idxmin()
    top_crop = crop_yield.idxmax()
    bot_crop = crop_yield.idxmin()
    top_feature = importance_df.iloc[0]["Feature"]
    top_importance = importance_df.iloc[0]["Importance"]

    insights = [
        f"1. REGIONAL DIFFERENCES: '{top_region}' achieves the highest average crop yield "
        f"({region_yield[top_region]:.2f}), while '{bot_region}' has the lowest "
        f"({region_yield[bot_region]:.2f}). Regional factors such as climate, soil type, "
        f"and farming practices likely drive this gap.",

        f"2. CROP TYPE DIFFERENCES: '{top_crop}' yields the highest average "
        f"({crop_yield[top_crop]:.2f}), while '{bot_crop}' yields the lowest "
        f"({crop_yield[bot_crop]:.2f}). Selecting appropriate crop varieties for local "
        f"conditions can significantly impact output.",

        f"3. FERTILISER AND YIELD: The correlation between Fertiliser_Use and Crop_Yield "
        f"is {fert_corr:.4f}. While fertiliser use has a relationship with yield, "
        f"the analysis also identified farms with high fertiliser use but below-average "
        f"yield, suggesting that more fertiliser alone does not guarantee better results.",

        f"4. IRRIGATION AND YIELD: Irrigation shows a correlation of {irr_corr:.4f} with "
        f"Crop_Yield. Some farms with high irrigation still record low yields, indicating "
        f"that over-irrigation or poor drainage can be counterproductive. Irrigation "
        f"management should consider soil type and crop requirements.",

        f"5. STRONGEST NUMERICAL CORRELATION: '{top_corr_feature}' has the strongest "
        f"correlation with Crop_Yield ({top_corr_value:.4f}), making it an important "
        f"factor to monitor and manage.",

        f"6. RANDOM FOREST MODEL ACCURACY: The model achieved an R2 of {r2:.4f}, meaning "
        f"approximately {r2*100:.1f}% of the variance in crop yield can be explained by "
        f"the available features. This demonstrates the predictive value of the collected "
        f"farm data.",

        f"7. MOST IMPORTANT PREDICTOR (Random Forest): '{top_feature}' is the most "
        f"important feature for predicting crop yield with an importance score of "
        f"{top_importance:.4f}. Farm managers and agronomists should prioritise "
        f"monitoring and optimising this variable.",
    ]

    for insight in insights:
        print(f"\n  {insight}")


# =============================================================================
# 13. PRACTICAL ACTION PLAN
# =============================================================================
def generate_action_plan(df, importance_df):
    """Print data-driven farming recommendations."""
    print_section(14, "Practical Action Plan")

    top_features = importance_df["Feature"].head(3).tolist()

    recommendations = [
        "1. OPTIMISE FERTILISER USAGE",
        "   - Conduct soil tests before applying fertiliser to match nutrient needs.",
        "   - Farms with high fertiliser use but low yield should audit application "
        "methods and timing.",
        "   - Consider slow-release or precision fertilisation techniques.",

        "\n2. IMPROVE IRRIGATION MANAGEMENT",
        "   - Use drip or precision irrigation instead of flood irrigation to reduce waste.",
        "   - Monitor soil moisture levels to avoid over-irrigation.",
        "   - High-irrigation, low-yield farms should evaluate drainage and water scheduling.",

        "\n3. CONSIDER SOIL CONDITIONS",
        "   - Soil properties are a key factor in yield. Invest in soil health through "
        "crop rotation, organic matter, and pH management.",
        "   - Match crop selection to existing soil characteristics.",

        "\n4. SELECT SUITABLE CROPS FOR EACH REGION",
        "   - Base crop selection on regional yield performance observed in this dataset.",
        f"  - Prioritise crop types that historically yield well in each specific region.",

        "\n5. USE RAINFALL AND TEMPERATURE DATA",
        "   - Align planting schedules with seasonal rainfall patterns.",
        "   - Use temperature data to select heat-tolerant or cold-tolerant varieties.",
        "   - Consider supplementary irrigation in low-rainfall years.",

        "\n6. FOCUS ON HIGH-IMPORTANCE FEATURES",
        f"  - The Random Forest model identified {', '.join(top_features)} as the top "
        f"predictors of crop yield.",
        "   - Allocate resources and monitoring efforts to these variables first.",

        "\n7. ADOPT PREDICTIVE MODELLING FOR YIELD PLANNING",
        "   - Use the trained Random Forest model (or similar) to forecast expected yields "
        "before the season starts.",
        "   - This can help allocate resources, plan logistics, and set realistic targets.",
    ]

    for line in recommendations:
        print(f"  {line}")


# =============================================================================
# MAIN – ORCHESTRATE ALL STEPS
# =============================================================================
def main():
    print("\n" + "=" * 50)
    print("   AGRICULTURE CROP YIELD ANALYSIS")
    print("   Why is Crop Yield Different?")
    print("=" * 50)

    # 1. Load
    df = load_data()

    # 2. Clean
    df = clean_data(df)

    # 3. EDA
    region_yield, crop_yield, corr_matrix = perform_eda(df)

    # 4. Visualizations
    print_section("4", "Visualizations")
    create_visualizations(df, corr_matrix)

    # 6. Fertilizer analysis
    fert_corr = analyze_fertilizer(df)

    # 7. Irrigation analysis
    irr_corr = analyze_irrigation(df)

    # 8. Inefficient farms
    identify_inefficient_farms(df)

    # 9-10. Train and evaluate model
    rf_model, X_test, y_test, feature_names = train_random_forest(df)
    mae, rmse, r2, y_pred = evaluate_model(rf_model, X_test, y_test)

    # 11. Feature importance
    importance_df = analyze_feature_importance(rf_model, feature_names)

    # 12. Key insights
    generate_insights(df, region_yield, crop_yield, corr_matrix,
                      importance_df, fert_corr, irr_corr, r2)

    # 13. Action plan
    generate_action_plan(df, importance_df)

    print("\n" + "=" * 50)
    print("   ANALYSIS COMPLETE")
    print(f"   Charts saved in  : outputs/visualizations/")
    print(f"   Model plots saved: outputs/model/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
