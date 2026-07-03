import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.active_learning import ActiveLearner, run_active_learning_cycle

np.random.seed(42)

n_samples = 1000
n_features = 50
n_classes = 2

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] > 0).astype(int)

n_train = 800
n_test = 200

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

n_initial = 50
indices = np.random.permutation(n_train)
X_initial = X_train[indices[:n_initial]]
y_initial = y_train[indices[:n_initial]]
X_pool = X_train[indices[n_initial:]]
y_pool = y_train[indices[n_initial:]]

print(f"Initial labeled: {len(X_initial)}, Pool: {len(X_pool)}, Test: {len(X_test)}")

model = LogisticRegression(max_iter=1000)
learner = ActiveLearner(estimator=model, strategy="random")
learner.fit(X_initial, y_initial)

results = run_active_learning_cycle(
    learner=learner,
    X_pool=X_pool,
    y_pool=y_pool,
    X_test=X_test,
    y_test=y_test,
    label_ratios=[0.1, 0.3, 0.5],
    initial_ratio=0.05,
    verbose=True,
)

print("\nRandom sampling results:")
for i, ratio in enumerate(results["label_ratios"]):
    print(f"Ratio: {ratio*100:.0f}% | Accuracy: {results['accuracies'][i]:.4f}")

model2 = LogisticRegression(max_iter=1000)
learner2 = ActiveLearner(estimator=model2, strategy="entropy")
learner2.fit(X_initial, y_initial)

results2 = run_active_learning_cycle(
    learner=learner2,
    X_pool=X_pool,
    y_pool=y_pool,
    X_test=X_test,
    y_test=y_test,
    label_ratios=[0.1, 0.3, 0.5],
    initial_ratio=0.05,
    verbose=True,
)

print("\nEntropy sampling results:")
for i, ratio in enumerate(results2["label_ratios"]):
    print(f"Ratio: {ratio*100:.0f}% | Accuracy: {results2['accuracies'][i]:.4f}")

print("\nTest completed successfully!")