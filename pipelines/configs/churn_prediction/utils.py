import numpy as np
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier

parameters_train_ada_boost_upsampling = {
    "AdaBoostClassifier__base_estimator": [
        DecisionTreeClassifier(
            max_depth=3,
            max_leaf_nodes=9,
            min_impurity_decrease=0.001,
            min_samples_leaf=8,
            random_state=1,
        ),
        DecisionTreeClassifier(
            max_depth=5,
            max_leaf_nodes=9,
            min_impurity_decrease=0.001,
            min_samples_leaf=10,
            random_state=1,
        ),
        DecisionTreeClassifier(
            max_depth=8,
            max_leaf_nodes=9,
            min_impurity_decrease=0.001,
            min_samples_leaf=8,
            random_state=1,
        ),
    ],
    "AdaBoostClassifier__n_estimators": randint(10, 110),
    "AdaBoostClassifier__learning_rate": np.arange(0.01, 0.10, 0.05),
}
