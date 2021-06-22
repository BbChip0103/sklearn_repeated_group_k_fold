# sklearn_repeated_group_k_fold

GroupKFold is done deterministically, it makes repeated GroupKFold impossible.
So this code have two features.
1. Add shuffle option like general KFold.
2. RepeatedGroupKFold like RepeatedKFold based on (1.)

---

``` python
import numpy as np
from sklearn.model_selection import RepeatedGroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 2, 2])
groups = np.array([0, 1, 2, 3])
rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=2652124)
for train_index, test_index in rkf.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
=== Output ===  
TRAIN: [1 2] TEST: [0 3]  
TRAIN: [0 3] TEST: [1 2]  
TRAIN: [0 1] TEST: [2 3]  
TRAIN: [2 3] TEST: [0 1]  

---

I tried to implement this in a scikit-learn compatible way.
Please find below source.

[Github source](https://github.com/BbChip0103/sklearn_repeated_group_k_fold/blob/main/sklearn_repeated_group_k_fold.py)
[Example in the colab](https://colab.research.google.com/drive/1OzRELL1vU15kEtA6OKTROz6ExekiJjL5?usp=sharing)

---

[Sklearn issue #20317](https://github.com/scikit-learn/scikit-learn/issues/20317)
