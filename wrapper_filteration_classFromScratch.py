import pandas as pd
import numpy as np

class SequentialFeatureSelector:
    def __init__(self, estimator, n_features_to_select, direction="forward", scoring="accuracy"):
        """
        Initialize the SequentialFeatureSelector
        estimater : sklearn estimater
            A scikit-learn compatible model with fit and score methods
        n_features_to_select : int
            Number of features to select
        direction : str, default="forward"
            "forward" for forward selection or "backward" for backward elimination
        scoring : str, default="accuracy"
            Scoring metric: "accuracy" for classification or "r2" for regression
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.selected_features_ = None
        self.support_ = None

    def fit(self, X, y):
        """
        Returns:
        self object
        """
        n_features = X.shape[1]
        
        if self.direction == "forward":
            self.selected_features_ = self._forward_selection(X, y, n_features)
        elif self.direction == "backward":
            self.selected_features_ = self._backward_elimination(X, y, n_features)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")
        
        # Create boolean support mask
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[self.selected_features_] = True
        
        return self

    def _forward_selection(self, X, y, n_features):
        """Forward selection: start with 0 features, add best feature at each step."""
        selected = []
        remaining = list(range(n_features))
        
        for _ in range(self.n_features_to_select):
            if not remaining:
                break
            
            best_score = -np.inf
            best_feature = None
            
            # Evaluate each remaining feature
            for i_fichr in remaining:
                # Combine selected features with current candidate
                features_to_use = selected + [i_fichr]
                #print(i_fichr)
                X_subset = X[:, features_to_use]
                
                # Fit and score
                self.estimator.fit(X_subset, y)
                score = self.estimator.score(X_subset, y)
                
                # Update  if this itis better
                if score > best_score:
                    best_score = score
                    best_feature = i_fichr
            
            # Add it the best feature to selected
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
        
        return np.array(selected)

    def _backward_elimination(self, X, y, n_features):
        "Backward elimination start with all features annd remove worstfeature at each step."
        selected = list(range(n_features))
        
        # Remove features until only n_features_to_select remain
        while len(selected) > self.n_features_to_select:
            worst_score = np.inf
            worst_ficher = None
            worst_idx_in_selected = None
            
            # Evaluate each selected feature
            for idx, feature_idx in enumerate(selected):
                # Try removing this feature
                features_to_use = [f for i, f in enumerate(selected) if i != idx]
                if not features_to_use:
                    continue
                
                X_subset = X[:, features_to_use]
                
                # Fit and score
                self.estimator.fit(X_subset, y)
                score = self.estimator.score(X_subset, y)
                
                # Update worst if this is worse (or less good)
                if score < worst_score:
                    worst_score = score
                    worst_ficher = feature_idx
                    worst_idx_in_selected = idx
            
            # Remove worst feature
            if worst_idx_in_selected is not None:
                selected.pop(worst_idx_in_selected)
        
        return np.array(selected)

    def transform(self, X):
        """
        Transform X to contain only selected features.
        X : arrayof shape (n_samples, n_features)Data to transform
        X_subset : array of shape (n_samples, n_features_to_selectData with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("fit() must be called before transform()")
        
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """
        Fit the selector and transform X.
        X_subset : array of shape (n_samples, n_features_to_select)Transformed data with only selected features
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.
        
        Parameters:
        -----------
        indices : bool, default=False
            If True, return the indices of selected features
            If False, return a boolean mask
        support : array
            Boolean mask or indices of selected features
        """
        if self.support_ is None:
            raise ValueError("fit() must be called before get_support()")
        
        if indices:
            return self.selected_features_
        return self.support_

























