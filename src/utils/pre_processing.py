import os
import pandas as pd
import numpy as np 

def extract_data(filename: str) -> tuple[pd.DataFrame, str]:

    ''' 
        Extract Data Function:
        - Checks file extention: Different extentions will imply different operations. 
        - Accepted formats: csv, json, excel, txt. 
        - TODO: fasta (Bioinformatics)
        
        Parameters: 
        - filename (str): The name of the file, designed in a way where the datafile should be in the same directory. 

        Returns: 
        - pd.Dataframe: dataframe based on specified filename given. 
        
    '''

    extention: str = os.path.splitext(filename)[1]

    if extention == '.csv': 
        
        data: pd.DataFrame = pd.read_csv(filename)

    elif extention == '.json':

        data: pd.DataFrame = pd.read_json(filename)

    elif extention == '.xls':

        data: pd.DataFrame = pd.read_excel(filename)

    elif extention == '.txt':

        with open(filename, 'r') as file:

            lines = file.readlines()
        
        data: pd.DataFrame = pd.DataFrame({'lines': lines})

    else: 

        print('File Extention Not Supported')

    return data

def dtype_separation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    '''
        Separates the numeric data with categorical data. 

        Parameters: 
            - data (pd.DataFrame): original population or sample dataset.

        Inconsistent: Doesnt always return proper divisins
            - May or may not fix. It works for now. 

        Returns: pd.Dataframes (x2): One is strictly numeric, other one is strictly categorical / string based.

    '''

    data_numerical: pd.DataFrame = data.select_dtypes(include=[np.number])

    data_categorical: pd.DataFrame = data.select_dtypes(exclude=[np.number])

    return data_numerical, data_categorical

def normalize(data: pd.DataFrame, abs: bool) -> pd.DataFrame:

    '''
        Normalizes the data. 

        Parameters: 
        - Data (pd.DataFrame): Original dataset. 
        - abs (bool): Absolute function indicator.

        Returns: 
        - Normalized data with the range of [-1,1].

    '''

    for col in data.columns: 

        column: pd.Series = data[col]

        if abs: 
            max_val_abs: float = column.abs().max()
            data[col] = ( column / max_val_abs )

        else:
            min_val: float = column.min()
            max_val: float = column.max()
            data[col] = ( (column - min_val) / (max_val - min_val) )

    return data

def standardize(data: pd.DataFrame) -> pd.DataFrame:

    '''
        The function that I have a difficuly correctly pronouncing my even in my head. 
        Calculates the values of points based on their z-score. 

        Z-score Fomula:
        z = (x[i] - mean) / std

        Parameters: 
        - data (pd.DataFrame): original dataset. 

        Returns:
        - pd.Dataframe: Standardized dataset 
    '''


    for col in data.columns: 

        column: pd.Series = data[col]
        col_std: float = column.std()
        col_mean: float = column.mean()

        data[col] = (column - col_mean) / col_std

    return data

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:

    '''
        Removes the outliers present int he dataset. Designed to handle the whole
        dataset instead of a pd.Series (single column). This method implements the 
        Inter Quartile Range (IQR) approach. Anything below (1.5 * q) of the quadrant 1 (Q1)
        and above (1.5 * IQR) of the quadrand 3 (Q3) gets removed as an outlier. 

        Parameters: 
        - (pd.DataFrame): Original dataset. 

        Returns: 
        - pd.DataFrame: Original dataset with outliers removed. 

        TODO Improvements: 
        - Implement 3 * Standard Deviation (std) option. 
            - Anything +/- (3 * std) from the mean are not making the cut. 
        
    '''


    for col in data.columns:

        column = data[col]
        q1, q3 = column.quantile(0.25), column.quantile(0.75)
        IQR = q3 - q1
        low, high = q1 - (1.5 * IQR), q3 + (1.5 * IQR)
        
        # Filter rows based on the column's outlier criteria
        data = data[(column >= low) & (column <= high)]
        
    return data

def pipeline(data: pd.DataFrame, pipeline: list) -> pd.DataFrame:

    '''
        Executes every function specified by the user in a list.

        Parameters: 
        - data (pd.DataFrame): Original dataset. 
        - pipeline (list): List of pre-processing functions. 

        Returns: 
        - pd.Datafame: Processed Dataset. 
    
    '''

    for operation in pipeline:

        if operation == 'RemoveNa':

            data.dropna(axis=0)

        elif operation == 'RemoveOutliers':

            data = remove_outliers(data)

        elif operation == 'Normalize':

            data = normalize(data, abs=False)
        
        elif operation == 'Normalize_Abs':

            data = normalize(data, abs=True)
        
        elif operation == 'Standardize':

            data = standardize(data)

        else:

            print(f'Invalid Operation: {operation}')

    return data