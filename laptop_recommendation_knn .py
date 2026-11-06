# Laptop Recommendation System using KNN

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# 1. Load dataset
data = pd.read_csv(r"C:\Users\manda\OneDrive\Desktop\ML Project\Predictions(laptop).csv")
data.columns = data.columns.str.strip()

# Handle different column names
if "ScreenRes" not in data.columns and "ScreenResolution" in data.columns:
    data.rename(columns={"ScreenResolution": "ScreenRes"}, inplace=True)

if "Price" not in data.columns and "Price_euros" in data.columns:
    data.rename(columns={"Price_euros": "Price"}, inplace=True)

columns = [
    "Company", "TypeName", "Inches", "ScreenRes", "Cpu",
    "Ram", "Memory", "Gpu", "OpSys", "Weight", "Price"
]

missing = [col for col in columns if col not in data.columns]
if missing:
    print("Missing columns:", missing)
    print("Available columns:", list(data.columns))
    raise SystemExit("Please check your CSV column names.")

data = data[columns].copy()

# 2. Clean numerical columns
data["Ram"] = pd.to_numeric(
    data["Ram"].astype(str).str.replace("GB", "", regex=False),
    errors="coerce"
)
data["Weight"] = pd.to_numeric(
    data["Weight"].astype(str).str.replace("kg", "", regex=False),
    errors="coerce"
)
data["Inches"] = pd.to_numeric(data["Inches"], errors="coerce")
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

data = data.dropna(subset=["Ram", "Weight", "Inches", "Price"])
data = data.reset_index(drop=True)

if data.empty:
    raise SystemExit("No valid laptop records found in the dataset.")

# 3. Get user requirements
print("\nLAPTOP RECOMMENDATION SYSTEM")
brand = input("Preferred brand : ").strip()
laptop_type = input("Laptop type : ").strip()
processor = input("Processor preference : ").strip()
storage = input("Storage preference : ").strip()

try:
    ram_text = input("Required RAM in GB (e.g., 8): ").strip()
    screen_text = input("Screen size in inches : ").strip()
    budget = float(input("Maximum budget: "))

    ram = float(ram_text) if ram_text else data["Ram"].median()
    screen = float(screen_text) if screen_text else data["Inches"].median()
except ValueError:
    raise SystemExit("Enter valid numbers for RAM, screen size, and budget.")

# 4. Filter by budget when possible
available = data[data["Price"] <= budget].copy()

if available.empty:
    print("No laptops found within this budget.")
    print("Showing recommendations from the full dataset instead.")
    available = data.copy()

# 5. Prepare a preference row
categorical = [
    "Company", "TypeName", "ScreenRes", "Cpu",
    "Memory", "Gpu", "OpSys"
]
numerical = ["Inches", "Ram", "Weight"]
features = categorical + numerical

query = {}

for col in categorical:
    common = available[col].dropna().astype(str).mode()
    query[col] = common.iloc[0] if not common.empty else ""

query["Inches"] = screen
query["Ram"] = ram
query["Weight"] = available["Weight"].median()

if brand:
    query["Company"] = brand

if laptop_type:
    query["TypeName"] = laptop_type

# Find the closest processor match in the dataset
if processor:
    matches = available[
        available["Cpu"].astype(str).str.contains(
            processor, case=False, na=False
        )
    ]
    if not matches.empty:
        query["Cpu"] = matches["Cpu"].mode().iloc[0]

# Find the closest storage match in the dataset
if storage:
    matches = available[
        available["Memory"].astype(str).str.contains(
            storage, case=False, na=False
        )
    ]
    if not matches.empty:
        query["Memory"] = matches["Memory"].mode().iloc[0]

# 6. Convert text to numbers and scale numerical features
preprocessor = ColumnTransformer([
    ("text", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("numbers", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numerical)
])

X = preprocessor.fit_transform(available[features])
query_data = pd.DataFrame([query], columns=features)
query_X = preprocessor.transform(query_data)

# 7. Find similar laptops using KNN
k = min(10, len(available))
model = NearestNeighbors(n_neighbors=k, metric="cosine")
model.fit(X)

distances, indexes = model.kneighbors(query_X)

# 8. Display the top five recommendations
result = available.iloc[indexes[0]].copy()
result["Distance"] = distances[0]
result = result.sort_values("Distance").head(5)

print("\nRECOMMENDED LAPTOPS")
print(result[
    ["Company", "TypeName", "Inches", "Cpu", "Ram",
     "Memory", "Gpu", "OpSys", "Weight", "Price"]
].to_string(index=False))

print("\nRecommendations are based on similarity to your preferences.")