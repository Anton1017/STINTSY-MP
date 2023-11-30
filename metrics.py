import numpy as np

def compute_rmse(predictions, actual):
    # write code here
    return np.sqrt((np.square((predictions - actual))).sum()/len(predictions))\

def compute_accuracy(predictions, actual):
    # write code here
    return round(sum(p == a for p, a in zip(predictions, actual))/len(actual) * 100, 3)

