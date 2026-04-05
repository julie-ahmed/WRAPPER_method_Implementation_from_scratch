from JulieAhmed_23011245 import SequentialFeatureSelector
from sklearn.datasets import load_breast_cancer;
from sklearn.linear_model import LogisticRegression;
import pandas as pd
print("hello1")
x,y=load_breast_cancer(return_X_y=True)
estimator = LogisticRegression(max_iter=1000)
print("hello2")
sfs = SequentialFeatureSelector(estimator, n_features_to_select=5, direction="forward", scoring="accuracy")
sfs.fit(x,y)
print("hello3")

print(sfs.get_support())
print("hello4")
print(sfs.selected_features_)
x_transformed = sfs.transform(x)
print(x_transformed.shape)
