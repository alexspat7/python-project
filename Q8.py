import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import mode

# --- Φόρτωση ---
df = pd.read_csv('crimes.csv')
continuous_features = ['hour_float', 'latitude', 'longitude', 'victim_age', 
                       'temp_c', 'humidity', 'dist_precinct_km', 'pop_density']
categorical_features = ['weapon_code', 'scene_type', 'weather', 'vic_gender']

df_train = df[df['split'] == 'TRAIN'].copy()
df_test = df[df['split'] == 'TEST'].copy()
X_train = df_train[continuous_features + categorical_features]
y_train = df_train['killer_id']
X_test = df_test[continuous_features + categorical_features]

# --- Preprocessing & PCA ---
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
X_train_proc = preprocessor.fit_transform(X_train)
pca = PCA(n_components=2).fit(X_train_proc)
X_train_pca = pca.transform(X_train_proc)

# --- K-Means Training ---
kmeans = KMeans(n_clusters=8, random_state=42).fit(X_train_pca)
train_labels = kmeans.labels_

# Mapping Clusters to Killers
mapping = {}
for i in range(8):
    mask = (train_labels == i)
    if np.sum(mask) > 0:
        mapping[i] = mode(y_train[mask], keepdims=True).mode[0]
    else:
        mapping[i] = 1 

# --- Prediction on TEST ---
X_test_proc = preprocessor.transform(X_test)
X_test_pca = pca.transform(X_test_proc)
test_clusters = kmeans.predict(X_test_pca)
test_preds = [mapping[c] for c in test_clusters]

plt.figure(figsize=(10, 8))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_preds, cmap='tab10', alpha=0.6)
plt.title('Q8: Unsupervised Predictions on TEST')
plt.savefig('q8_test_predictions.png')
print("Saved q8_test_predictions.png")