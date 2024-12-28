import numpy as np
import pandas as pd
import src.models as models
import src.utils.sampling as sp
import src.utils.pre_processing as pr
import src.utils.feature_engineering as fe

def main():

    # Extract Data: 
    data: pd.DataFrame = pr.extract_data('data/processed/data.csv')
    
    # Data Type Split: Numerical vs Categorical
    data_numerical, data_categorical = pr.dtype_separation(data)

    # Create Pipeline: 
    data_processed = pr.pipeline(data_numerical, pipeline=['RemoveNa', 
                                                           'RemoveOutliers', 
                                                           'Normalize'
                                                           ])

    # Sample data for hierarchial clustering:
    data_sample: pd.DataFrame = sp.random_sampling(data_processed, n_samples=30, replacement=False)
    

if __name__ == '__main__':

    main()