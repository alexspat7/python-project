import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Φόρτωση ---
df = pd.read_csv('crimes.csv')
continuous_features = ['hour_float', 'latitude', 'longitude', 'victim_age', 
                       'temp_c', 'humidity', 'dist_precinct_km', 'pop_density']
categorical_features = ['weapon_code', 'scene_type', 'weather', 'vic_gender']

df_train = df[df['split'] == 'TRAIN'].copy()
df_val = df[df['split'] == 'VAL'].copy()

X_train = df_train[continuous_features + categorical_features]
y_train = df_train['killer_id']
X_val = df_val[continuous_features + categorical_features]
y_val = df_val['killer_id']

# --- Pipeline ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([('pre', preprocessor), ('clf', RidgeClassifier())])
model.fit(X_train, y_train)

# --- Αξιολόγηση ---
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Linear Classifier Accuracy: {acc:.4f}")

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap='Reds')
plt.title('Q4: Linear CM')
plt.savefig('q4_confusion_matrix.png')
print("Saved q4_confusion_matrix.png")