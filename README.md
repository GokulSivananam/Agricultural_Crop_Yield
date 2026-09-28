# Agriculture Crop Yield Analysis
### Why is Crop Yield Different?

---

## Project Objective

An agricultural analytics company collects data from farms across different regions. The goal is to **analyse why crop yield differs** between farms, regions, and crop varieties, and to **identify the factors associated with higher crop yield** using data science techniques.

---

## Problem Statement

Crop yield variability is a major challenge for agricultural planning. Farmers and agribusinesses need to understand which factors — rainfall, irrigation, fertiliser use, soil quality, or temperature — most strongly influence how much a crop produces. This project uses exploratory data analysis and machine learning to answer those questions from real farm data.

---

## Dataset Description

**File:** `data/agriculture.csv`

| Column | Description |
|---|---|
| `Farm_ID` | Unique identifier for each farm |
| `Region` | Geographic region of the farm |
| `Crop_Type` | Type of crop grown |
| `Rainfall` | Rainfall received (mm or relative measure) |
| `Soil_Properties` | Soil quality index |
| `Irrigation` | Amount of irrigation applied |
| `Fertiliser_Use` | Amount of fertiliser used |
| `Temperature` | Average temperature during the growing season |
| `Crop_Yield` | **Target variable** — output yield of the crop |

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib | Base plotting library |
| Seaborn | Statistical visualizations |
| Scikit-learn | Machine learning (Random Forest, metrics, encoding) |
| VS Code | Development environment |

---

## Project Structure

```
Agriculture_Crop_Yield_Analysis/
│
├── data/
│   └── agriculture.csv            ← Place your dataset here
│
├── outputs/
│   ├── visualizations/
│   │   ├── yield_by_region.png
│   │   ├── yield_by_crop.png
│   │   ├── fertiliser_vs_yield.png
│   │   ├── irrigation_vs_yield.png
│   │   └── correlation_heatmap.png
│   └── model/
│       ├── actual_vs_predicted.png
│       └── feature_importance.png
│
├── agriculture_analysis.py        ← Main analysis script
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## Project Workflow

```
Load CSV → Clean Data → Exploratory Data Analysis
    → Visualisations → Fertiliser Analysis → Irrigation Analysis
        → Identify Inefficient Farms → Train Random Forest
            → Evaluate Model → Feature Importance
                → Key Insights → Action Plan
```

---

## Data Preprocessing

- Load `agriculture.csv` using Pandas.
- Strip whitespace from column names.
- Validate that all required columns are present; exit with a clear error if not.
- Check and report missing values; fill numerical columns with **median**, categorical columns with **mode**.
- Detect and remove duplicate rows.
- Check for invalid (negative) values in numerical columns.
- Reset index after cleaning.

---

## Exploratory Data Analysis

The EDA covers:

- Crop yield distribution
- Average yield by region
- Average yield by crop type
- Relationship between rainfall and crop yield
- Relationship between soil properties and crop yield
- Relationship between irrigation and crop yield
- Relationship between fertiliser use and crop yield
- Relationship between temperature and crop yield
- Full correlation matrix for all numerical variables

---

## Visualizations

| Chart | File | Description |
|---|---|---|
| Average Crop Yield by Region | `yield_by_region.png` | Bar chart comparing regional average yields |
| Crop Yield by Crop Type | `yield_by_crop.png` | Box plot showing yield spread per crop |
| Fertiliser Use vs Crop Yield | `fertiliser_vs_yield.png` | Scatter plot with crop type as hue |
| Irrigation vs Crop Yield | `irrigation_vs_yield.png` | Scatter plot with region as hue |
| Correlation Heatmap | `correlation_heatmap.png` | Lower-triangle heatmap of all numerical features |
| Actual vs Predicted | `actual_vs_predicted.png` | Model evaluation scatter plot |
| Feature Importance | `feature_importance.png` | Horizontal bar chart of RF importances |

---

## Random Forest Model

- **Algorithm:** `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Target:** `Crop_Yield`
- **Features:** `Region`, `Crop_Type`, `Rainfall`, `Soil_Properties`, `Irrigation`, `Fertiliser_Use`, `Temperature`
- **Encoding:** `LabelEncoder` applied to `Region` and `Crop_Type`
- **Train/Test Split:** 80% / 20% with `random_state=42`

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average absolute difference between actual and predicted |
| RMSE | Root Mean Squared Error — penalises larger errors more heavily |
| R² Score | Proportion of variance in yield explained by the model |

All values are calculated from test set predictions; none are hardcoded.

---

## Key Insights

Insights are automatically generated from the data and include:

1. Regional yield differences and which region performs best/worst
2. Crop type differences — which crops yield most/least
3. Fertiliser–yield relationship (correlation + farms with high input, low output)
4. Irrigation–yield relationship (same pattern analysis)
5. Strongest numerical correlate with crop yield
6. Random Forest model accuracy (R² score interpretation)
7. Most important predictor feature from the model

---

## Farming Action Plan

Data-driven recommendations are generated automatically, covering:

- Fertiliser optimisation (soil testing, precision application)
- Irrigation management (drip irrigation, soil moisture monitoring)
- Soil health improvement
- Crop-region matching based on observed yield data
- Rainfall and temperature-aligned planting schedules
- Focus areas based on top Random Forest feature importances
- Adopting predictive modelling for yield planning

---

## How to Install Dependencies

Make sure Python 3 is installed. Then open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

1. Place your dataset at `data/agriculture.csv`.
2. Open a terminal in the `Agriculture_Crop_Yield_Analysis/` folder.
3. Run:

```bash
python agriculture_analysis.py
```

All charts will be saved automatically to the `outputs/` folder.

---

## Notes

- The script validates that all required columns exist before running. If a column is missing, it will display a clear error and exit.
- No values in the output are hardcoded — all statistics, thresholds, and model metrics are calculated live from the dataset.
- Correlation is reported where relevant, with reminders that **correlation does not prove causation**.
