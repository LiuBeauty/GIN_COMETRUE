import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
y_pred = [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2]
y_true = [0, 0, 4, 0, 2, 0, 2, 0, 0, 0, 1]
acc = metrics.accuracy_score(y_true, y_pred)
print(acc)