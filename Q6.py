import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

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

# --- MLP Model ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

mlp_model = Pipeline([
    ('pre', preprocessor), 
    ('clf', MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))
])
mlp_model.fit(X_train, y_train)

acc = accuracy_score(y_val, mlp_model.predict(X_val))
print(f"MLP Accuracy: {acc:.4f}")

# --- Feature Importance ---
baseline_acc = acc
importances = {}
for col in X_val.columns:
    X_val_perm = X_val.copy()
    X_val_perm[col] = np.random.permutation(X_val_perm[col].values)
    drop = baseline_acc - accuracy_score(y_val, mlp_model.predict(X_val_perm))
    importances[col] = drop

sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
plt.figure(figsize=(8, 5))
plt.barh([x[0] for x in sorted_feats], [x[1] for x in sorted_feats], color='salmon', edgecolor='black')
plt.gca().invert_yaxis()
plt.title('Q6: Top 5 Feature Importance')
plt.savefig('q6_feature_importance.png')
print("Saved q6_feature_importance.png")