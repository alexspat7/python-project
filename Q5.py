import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

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

# --- SVM Model ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Best params: C=1, gamma=0.1
svm_model = Pipeline([
    ('pre', preprocessor),
    ('clf', SVC(C=1, gamma=0.1, kernel='rbf', probability=True))
])
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"SVM Accuracy: {acc:.4f}")

# --- Plots ---
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap='Greens')
plt.title('Q5: SVM CM')
plt.savefig('q5_confusion_matrix.png')

# Visualization Support Vectors
X_train_proc = preprocessor.fit_transform(X_train)
pca_vis = PCA(n_components=2).fit(X_train_proc)
X_train_pca = pca_vis.transform(X_train_proc)

sv_indices = svm_model.named_steps['clf'].support_
plt.figure(figsize=(10, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab10', alpha=0.1, s=10)
plt.scatter(X_train_pca[sv_indices, 0], X_train_pca[sv_indices, 1], c='none', edgecolors='black', marker='D', s=30, label='Support Vectors')
plt.title('Q5: SVM Support Vectors (PCA Space)')
plt.legend()
plt.savefig('q5_svm_vis.png')
print("Saved plots for Q5.")