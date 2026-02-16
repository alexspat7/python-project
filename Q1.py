import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# --- Φόρτωση Δεδομένων ---
df = pd.read_csv('crimes.csv')
df_train = df[df['split'] == 'TRAIN'].copy()

# --- Q1A: Ιστογράμματα ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cols = ['hour_float', 'victim_age', 'latitude', 'longitude']
for i, col in enumerate(cols):
    sns.histplot(df[df['split'] != 'TEST'][col], bins=30, ax=axes.flatten()[i], color='skyblue', edgecolor='black')
    axes.flatten()[i].set_title(f'Histogram of {col}')
plt.tight_layout()
plt.savefig('q1_histograms.png')
print("Saved q1_histograms.png")

# --- Q1B: GMM Analysis ---
X_hour = df_train['hour_float'].values.reshape(-1, 1)
mu_h, std_h = norm.fit(X_hour)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X_hour)
x_ax = np.linspace(0, 24, 1000).reshape(-1, 1)

plt.figure(figsize=(10, 6))
sns.histplot(X_hour, bins=50, stat='density', alpha=0.3, element="step", label='Data')
plt.plot(x_ax, norm.pdf(x_ax, mu_h, std_h), 'r--', linewidth=2, label='Single Gaussian')
plt.plot(x_ax, np.exp(gmm.score_samples(x_ax)), 'b-', linewidth=2, label='GMM (3 components)')
plt.title('Q1: Modelling Time of Day')
plt.legend()
plt.savefig('q1_gmm_analysis.png')
print("Saved q1_gmm_analysis.png")

# --- Q1C: 2D Pattern ---
plt.figure(figsize=(10, 6))
plt.scatter(df_train['hour_float'], df_train['latitude'], alpha=0.1, s=5, c='darkblue')
plt.title('Q1: Hour vs Latitude Pattern')
plt.xlabel('Hour')
plt.ylabel('Latitude')
plt.savefig('q1_2d_pattern.png')
print("Saved q1_2d_pattern.png")