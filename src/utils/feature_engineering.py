import numpy as np
import pandas as pd

'''
    TODO:
        : Fix PCA. (At some point)
        : Wrapping: Additive / Divisive. 
        : Lasso Regression
        : Filtering. 
        : Categorical Data Conversion: Label Encoding, One Hot Encoder
        
'''

# PCA implementaton using the SVD method. 
class PCA:

    '''
        This code should be studied, as its the primary factor to my demise. 
        This code may but most likely will never be fixed. 
    '''

    def __init__(self, n_components=None):
        
        self.n_components: int = n_components
        self.components: np.ndarray = None
        self.variance = None
        self.variance_ratio = None
        self.mean_ = None

    def fit(self, data: pd.DataFrame) -> None: 

        # Center Data with respect to the mean. 
        self.mean_ = data.mean(axis=0).values
        data_cntr = data - self.mean_

        # SVD:
        # U - Not used. 
        # S - Values related to explained variance 
        # Vt - Principle Components 
        U, S, Vt = np.linalg.svd(data_cntr, full_matrices=False)

        # Store PC's 
        self.components = Vt[: self.n_components]

        # Compute variance and variance ratio. 
        total_variance: float = (S ** 2).sum()
        explained_variance = (S ** 2) / (len(data) - 1)
        self.variance = explained_variance[:self.n_components]
        self.variance_ratio = self.variance / total_variance

    def transform(self, data: pd.DataFrame): 

        if self.components is None:

            raise ValueError("PCA is not fitted yet. Please call fit() first.")
        
        # Center the data and project onto principal components
        data_cntr = data - self.mean_

        return np.dot(data_cntr, self.components.T)
    
    def fit_transform(self, data: pd.DataFrame):

        self.fit(data)
        
        return self.transform(data)
    


