import numpy as np
from typing import Callable, Tuple
from sklearn.base import BaseEstimator
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.disagreement import max_disagreement_sampling, consensus_entropy_sampling, vote_entropy_sampling


def random_sampling(classifier: BaseEstimator, X: np.ndarray, n_instances: int = 1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    query_idx = np.random.choice(len(X), size=n_instances, replace=False)
    return query_idx, X[query_idx]


sampling_strategies = {
    "random": random_sampling,
    "entropy": entropy_sampling,
    "margin": margin_sampling,
    "uncertainty": uncertainty_sampling,
    "max_disagreement": max_disagreement_sampling,
    "consensus_entropy": consensus_entropy_sampling,
    "vote_entropy": vote_entropy_sampling,
}


def create_active_learner(
    estimator: BaseEstimator,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    strategy: str = "random",
) -> ActiveLearner:
    query_strategy = sampling_strategies.get(strategy, random_sampling)
    learner = ActiveLearner(
        estimator=estimator,
        query_strategy=query_strategy,
        X_training=X_initial,
        y_training=y_initial,
    )
    return learner


def create_committee(
    estimators: list,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    strategy: str = "max_disagreement",
) -> Committee:
    learners = [
        ActiveLearner(
            estimator=estimator,
            query_strategy=sampling_strategies["random"],
            X_training=X_initial,
            y_training=y_initial,
        )
        for estimator in estimators
    ]
    
    query_strategy = sampling_strategies.get(strategy, max_disagreement_sampling)
    committee = Committee(
        learner_list=learners,
        query_strategy=query_strategy,
    )
    return committee


def run_active_learning_cycle(
    learner: ActiveLearner,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_ratios: list = [0.1, 0.3, 0.5],
    initial_ratio: float = 0.05,
    verbose: bool = True,
) -> dict:
    n_samples = len(X_pool) + len(learner.X_training)
    results = {
        "accuracies": [],
        "f1_scores": [],
        "precision_scores": [],
        "recall_scores": [],
    }
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    current_labeled = len(learner.X_training)
    
    for ratio in label_ratios:
        target_labeled = int(n_samples * ratio)
        
        if target_labeled <= current_labeled:
            y_pred = learner.predict(X_test)
            results["accuracies"].append(accuracy_score(y_test, y_pred))
            results["f1_scores"].append(f1_score(y_test, y_pred))
            results["precision_scores"].append(precision_score(y_test, y_pred))
            results["recall_scores"].append(recall_score(y_test, y_pred))
            continue
        
        n_to_label = target_labeled - current_labeled
        
        if len(X_pool) >= n_to_label:
            query_idx, _ = learner.query(X_pool, n_instances=n_to_label)
            
            learner.teach(
                X=X_pool[query_idx],
                y=y_pool[query_idx]
            )
            
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
            current_labeled = len(learner.X_training)
        
        y_pred = learner.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        results["accuracies"].append(acc)
        results["f1_scores"].append(f1)
        results["precision_scores"].append(prec)
        results["recall_scores"].append(rec)
        
        if verbose:
            print(f"Ratio: {ratio*100:.0f}% | Labeled: {current_labeled} | Accuracy: {acc:.4f} | F1: {f1:.4f}", flush=True)
    
    return results