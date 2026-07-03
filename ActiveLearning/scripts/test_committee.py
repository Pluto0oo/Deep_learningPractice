import numpy as np
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import max_disagreement_sampling
from modAL.uncertainty import entropy_sampling

np.random.seed(42)

X = np.random.randn(1000, 50)
y = (X[:, 0] > 0).astype(int)

n_initial = 50
indices = np.random.permutation(len(X))
X_initial = X[indices[:n_initial]]
y_initial = y[indices[:n_initial]]
X_pool = X[indices[n_initial:]]
y_pool = y[indices[n_initial:]]
X_test = X[:100]
y_test = y[:100]

estimators = [LogisticRegression(max_iter=1000) for _ in range(3)]
learners = [
    ActiveLearner(
        estimator=estimator,
        query_strategy=entropy_sampling,
        X_training=X_initial,
        y_training=y_initial,
    )
    for estimator in estimators
]

committee = Committee(
    learner_list=learners,
    query_strategy=max_disagreement_sampling,
)

print("=== Testing Committee ===")
print(f"Number of learners: {len(committee.learner_list)}")
print(f"Has X_training: {hasattr(committee, 'X_training')}")
print(f"X_training value: {getattr(committee, 'X_training', 'N/A')}")
print(f"First learner X_training shape: {committee.learner_list[0].X_training.shape}")

try:
    query_idx, _ = committee.query(X_pool, n_instances=10)
    print(f"\nQuery successful! Indices shape: {query_idx.shape}")
    
    committee.teach(
        X=X_pool[query_idx],
        y=y_pool[query_idx]
    )
    print(f"Teach successful!")
    print(f"First learner X_training shape after teach: {committee.learner_list[0].X_training.shape}")
    
    y_pred = committee.predict(X_test)
    print(f"Predict successful! y_pred shape: {y_pred.shape}")
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
