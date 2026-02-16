import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

# --- Preprocessing & Training SVM (for coloring) ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
X_train_proc = preprocessor.fit_transform(X_train)
svm_model = SVC(kernel='rbf', C=1, gamma=0.1).fit(X_train_proc, y_train)

# --- PCA ---
pca = PCA(n_components=2).fit(X_train_proc)

# Scree Plot
pca_full = PCA().fit(X_train_proc)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca_full.explained_variance_) + 1), pca_full.explained_variance_, 'o-')
plt.title('Q7: Scree Plot')
plt.xlabel('Component')
plt.ylabel('Variance')
plt.savefig('q7_scree_plot.png')

# 2D Projection
X_val_proc = preprocessor.transform(X_val)
X_val_pca = pca.transform(X_val_proc)
y_pred_svm = svm_model.predict(X_val_proc)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_pred_svm, cmap='tab10', alpha=0.6)
plt.legend(*scatter.legend_elements(), title="Pred Killer (SVM)")
plt.title('Q7: PCA Projection of VAL')
plt.savefig('q7_pca_scatter.png')
print("Saved plots for Q7.")