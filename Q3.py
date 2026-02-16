import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# --- Φόρτωση ---
df = pd.read_csv('crimes.csv')
df_train = df[df['split'] == 'TRAIN'].copy()
df_val = df[df['split'] == 'VAL'].copy()
continuous_features = ['hour_float', 'latitude', 'longitude', 'victim_age', 
                       'temp_c', 'humidity', 'dist_precinct_km', 'pop_density']
killers = sorted(df_train['killer_id'].unique())

# --- Επανυπολογισμός MLE (Απαραίτητο για το Bayes) ---
killer_models = {}
for k in killers:
    X_k = df_train[df_train['killer_id'] == k][continuous_features].values
    mu_k = np.mean(X_k, axis=0)
    sigma_k = np.cov(X_k, rowvar=False) + np.eye(len(continuous_features)) * 1e-4
    killer_models[k] = {'mu': mu_k, 'sigma': sigma_k}

priors = {k: len(df_train[df_train['killer_id'] == k]) / len(df_train) for k in killers}

# --- Πρόβλεψη ---
def predict_bayes(X, models, priors, killers):
    preds = []
    for x in X:
        posteriors = []
        for k in killers:
            log_lik = multivariate_normal.logpdf(x, mean=models[k]['mu'], cov=models[k]['sigma'])
            posteriors.append(log_lik + np.log(priors[k]))
        preds.append(killers[np.argmax(posteriors)])
    return np.array(preds)

y_pred = predict_bayes(df_val[continuous_features].values, killer_models, priors, killers)
acc = accuracy_score(df_val['killer_id'], y_pred)
print(f"Gaussian Bayes Accuracy: {acc:.4f}")

# --- Plot ---
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(df_val['killer_id'], y_pred, display_labels=killers, cmap='Blues')
plt.title(f'Q3: Gaussian Bayes CM (Acc: {acc:.3f})')
plt.savefig('q3_confusion_matrix.png')
print("Saved q3_confusion_matrix.png")