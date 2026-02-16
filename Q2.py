import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# --- Φόρτωση Δεδομένων ---
df = pd.read_csv('crimes.csv')
df_train = df[df['split'] == 'TRAIN'].copy()
continuous_features = ['hour_float', 'latitude', 'longitude', 'victim_age', 
                       'temp_c', 'humidity', 'dist_precinct_km', 'pop_density']
killers = sorted(df_train['killer_id'].unique())

# --- Υπολογισμός MLE ---
killer_models = {}
for k in killers:
    X_k = df_train[df_train['killer_id'] == k][continuous_features].values
    mu_k = np.mean(X_k, axis=0)
    # Προσθήκη jitter για αποφυγή singular matrix
    sigma_k = np.cov(X_k, rowvar=False) + np.eye(len(continuous_features)) * 1e-4
    killer_models[k] = {'mu': mu_k, 'sigma': sigma_k, 'X': X_k}

# --- Heatmaps ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, k in enumerate(killers[:4]):
    sns.heatmap(killer_models[k]['sigma'], ax=axes.flatten()[i], cmap='coolwarm', center=0)
    axes.flatten()[i].set_title(f'Killer {k} Covariance')
plt.tight_layout()
plt.savefig('q2_heatmaps.png')
print("Saved q2_heatmaps.png")

# --- Ellipses ---
def draw_ellipse(mu, sigma, x_idx, y_idx, ax, color):
    cov_2d = np.array([[sigma[x_idx, x_idx], sigma[x_idx, y_idx]], 
                       [sigma[y_idx, x_idx], sigma[y_idx, y_idx]]])
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 3 * np.sqrt(vals)
    ell = Ellipse(xy=(mu[x_idx], mu[y_idx]), width=width, height=height, angle=theta, 
                  edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(killers)))
lat_idx, long_idx = 1, 2 # Indices for lat/long

for i, k in enumerate(killers):
    X_k = killer_models[k]['X']
    ax.scatter(X_k[:, lat_idx], X_k[:, long_idx], s=5, alpha=0.3, color=colors[i])
    draw_ellipse(killer_models[k]['mu'], killer_models[k]['sigma'], lat_idx, long_idx, ax, colors[i])

plt.title('Q2: MLE Ellipses (Latitude vs Longitude)')
plt.savefig('q2_ellipses.png')
print("Saved q2_ellipses.png")