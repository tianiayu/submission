import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# 1. LOAD DATASET
DATA_PATH = 'D:\Bootcamp\Laskar AI\Submission\Eksperimen_SML_Tiani\housing.csv'
df = pd.read_csv(DATA_PATH)

# 2. FEATURE SETUP
target = 'price'
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']

X = df.drop(columns=[target])
y = df[target]

# 3. PREPROCESSING PIPELINE
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# 4. TRANSFORM DATA
X_processed = pipeline.fit_transform(X)

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 6. SAVE OUTPUT
# Convert back to DataFrame
train_df = pd.DataFrame(X_train)
train_df['price'] = y_train.values

test_df = pd.DataFrame(X_test)
test_df['price'] = y_test.values

# Create output folder
OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save to CSV
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print("Preprocessing completed and files saved successfully.")
