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