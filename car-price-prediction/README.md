## problem statement
predicting the selling price of a used car.

Input:
car’s brand, model, year, mileage, fuel type, engine, transmission, colors, accident history, etc.

Output:
the target variable you want the ML model to learn = price.

# Why predict price?
Because in real life:

Car dealers want to estimate market value

Platforms like Cars24, CarDekho, OLX Autos do automated valuation

Sellers want guidance for listing price

Buyers want to avoid overpaying.

Machine learning helps build a valuation model that takes a car description → returns estimated market price.

## Inference of the dataset after describe(include='all').T

1. The dataset contains 4009 used car listings across 57 unique brands. Ford is the most common brand with 386 listings.
2. The model column has very high cardinality (1898 unique models), indicating that direct one-hot encoding is not practical.
3. The model_year ranges from 1974 to 2024 with a mean of 2015, suggesting a wide age range among cars.
4. Mileage values are stored as strings (e.g., “110,000 mi.”); these need cleaning to convert to numeric.
5. Fuel type is dominated by “Gasoline” (3309 listings), but 7 fuel types exist.
6. Engine column contains complex strings; numeric engine capacity must be extracted.
7. Transmission has 62 unique values but mostly revolves around A/T; can be simplified.
8. Exterior and interior colors show high cardinality; colors can be grouped.
9. The accident column has two categories; useful for binary encoding.
10. Clean_title has only one category (Yes), providing no variance; should be dropped.

## preprocessing column-by-column.

# Numeric columns

model_year, mileage, engine_numeric, price

Preprocessing:

Convert all to numeric

Fix skew (log transform)

StandardScaler if using LR/KNN/SVM

No scaling for tree models

# categorical (low cardinality)

fuel_type, transmission, accident

Preprocessing:

One-hot encoding

# Categorical (high cardinality)

brand, ext_col, int_col

Preprocessing:

Group rare categories → “Other Brand”

Reduce categories

One-hot encode OR target encode

# Very high-cardinality text columns

model (1898 unique), engine (complex string patterns)

Preprocessing:

Extract sub-features

Do NOT one-hot encode entire column

# Columns to drop

clean_title = only one unique value
→ Drop completely.

------------------------------------------------------------------------------------

- Used Car Price Prediction – Machine Learning Project
- Problem Statement
    The goal of this project is to predict the selling price of a used car based on its characteristics such as brand, age, mileage, fuel type, engine details, transmission, colors, and accident history.

- Inputs
    Brand
    Model year
    Mileage
    Fuel type
    Engine details
    Transmission
    Exterior & interior color
    Accident history
    Clean title indicator

- Output

Target variable: price (selling price of the used car)

- Why Predict Used Car Prices?

In real-world applications:

    Car dealers want to estimate fair market value

    Platforms like Cars24, CarDekho, OLX Autos use automated pricing systems

    Sellers want guidance for listing price

    Buyers want to avoid overpaying

    Machine Learning helps build a valuation system that maps
    car attributes → estimated market price

- Dataset Overview

Rows: 4009 used car listings

Brands: 57 unique brands

- Target: Selling price (price)

- Initial Data Understanding (describe(include='all').T)

    - Key insights from dataset inspection:

    Dataset contains 4009 records across 57 brands

    Ford is the most frequent brand (386 listings)

    model column has very high cardinality (1898 unique values) → not suitable for direct one-hot encoding

    model_year ranges from 1974 to 2024, showing wide age diversity

    milage stored as strings (e.g., "110,000 mi.") → required cleaning

    fuel_type is dominated by Gasoline, but multiple fuel categories exist

    engine column contains complex text → numeric engine capacity extracted

    transmission has many variants but mostly revolves around Automatic

    Exterior & interior colors have high cardinality → simplified into groups

    clean_title has only one unique value → provides no information and was dropped

-  Data Cleaning & Preprocessing
    - Numerical Features
        model_year
        milage
        engine_num
        price
    - Steps applied:
        Converted all to numeric
        Handled skewed distributions using log transformation
    - Created derived features:
        car_age
        milage_log
        price_log

Scaling applied where required (for linear models)

- Categorical Features (Low Cardinality)
    fuel_type
    transmission_simple
    accident_flag
    - Processing:
        Cleaned inconsistent labels
        One-hot encoded

- Categorical Features (High Cardinality)
    brand
    ext_col
    int_col
    - Processing:
        Grouped rare categories into "Other"
        Reduced dimensionality
        One-hot encoded after grouping

- High Cardinality Text Features
    model
    engine (raw text)

- Approach:
    Avoided direct encoding
    Extracted meaningful numeric features
    Dropped raw text columns

- Dropped Columns
    clean_title → only one unique value

- Exploratory Data Analysis (EDA)
    - Univariate Analysis
        Distribution analysis for numerical variables
        Identified skewness, outliers, and data ranges
        Bivariate Analysis
        Scatter plots and box plots to study relationships with price

- Key observations:
    Car age ↑ → Price ↓
    Mileage ↑ → Price ↓
    Accident history lowers resale value
    Fuel type & transmission moderately influence price

- Modeling Approach
    Feature Set Used - 
        brand, car_age, milage_log, engine_num,
        fuel_type, transmission_simple,
        clean_title_flag, accident_flag,
        ext_col_simple, int_col_simple

- Target - price_log

- Models Trained & Evaluated
    Model	                    RMSE	R² Score
    Linear Regression	        ~0.46	~0.68
    Random Forest (Baseline)    ~0.41	~0.75
    Random Forest (Tuned)	    ~0.41	~0.74
    Gradient Boosting	        ~0.42	~0.73
    XGBoost (Best)	            ~0.39	~0.77

- Best Model
    XGBoost Regressor
    Captures non-linear relationships
    Explains ~77% variance in used car prices
    Lowest RMSE among all models

- Model Saving & Reusability
    Final model saved as a pipeline
    Includes preprocessing + model
    Ready for deployment in production / Streamlit app