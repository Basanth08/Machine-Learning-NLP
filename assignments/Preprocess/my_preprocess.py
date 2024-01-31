import numpy as np
from copy import deepcopy

class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        self.norm = norm
        self.axis = axis
        self.offsets = None
        self.scalers = None

    def fit(self, X):
        X_array  = np.asarray(X)
        m, n = X_array.shape
        if self.axis == 1:
            self.offsets = []
            self.scalers = []
            self.offsets, self.scalers = zip(*[self.normalize_vectors(X_array[:, col]) for col in range(n)])
        
        elif self.axis == 0:
            self.offsets = []
            self.scalers = []
            self.offsets, self.scalers = zip(*[self.normalize_vectors(X_array[row]) for row in range(m)])
        
        else:
            raise Exception("Unknown axis.")

    def transform(self, X):
        X_norm = deepcopy(np.asarray(X))
        m, n = X_norm.shape
        if self.axis == 1:
            if n != len(self.offsets) or n != len(self.scalers):
                raise ValueError("Incompatible shapes for element-wise division along columns.")
            X_norm = (X_norm - np.array(self.offsets)) / np.array(self.scalers)
        
        elif self.axis == 0:
            if m != len(self.offsets) or m != len(self.scalers):
                raise ValueError("Incompatible shapes for element-wise division along rows.")
            offsets_array = np.array(self.offsets).reshape(-1, 1)
            scalers_array = np.array(self.scalers).reshape(-1, 1)
            X_norm = (X_norm - offsets_array) / scalers_array
        
        else:
            raise Exception("Unknown axis.")
        return X_norm

    def normalize_vectors(self, x):
        if self.norm == "Min-Max":
            offset = np.min(x)
            scaler = np.max(x) - np.min(x)   
        
        elif self.norm == "L1":
            offset = 0
            scaler = np.sum(np.abs(x))
        
        elif self.norm == "L2": 
            offset = 0
            scaler = np.linalg.norm(x, ord=2)
        
        elif self.norm == "Standard_Score":
            offset = np.mean(x)  
            scaler = np.std(x)
        else:
            raise Exception("Unknown normalization")
        return offset, scaler

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(labels, ratio, replace=True):
    if ratio <= 0 or ratio >= 1:
        raise Exception("Ratio must be between 0 and 1.")
        
    labels_arr = np.asarray(labels) 
    classes, class_counts = np.unique(labels_arr, return_counts=True)
    samples_per_class = np.ceil(class_counts * ratio).astype(int)
    indices = []
    
    for cls, num_samples in zip(classes, samples_per_class):
        cls_indices = np.where(labels_arr == cls)[0]
        if replace:
            sampled = np.random.choice(cls_indices, num_samples, replace=True)
        else:
            num_samples = min(num_samples, len(cls_indices))
            sampled = np.random.choice(cls_indices, num_samples, replace=False) 
        indices.extend(sampled)
    return np.array(indices)
