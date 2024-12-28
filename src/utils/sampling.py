import numpy as np 
import pandas as pd 
import random as rd

'''
    TODO: 
        : Bagging (Bootstrap Aggregating) (class or function?)
        : AdaBoost (class or function?)
          
'''

def random_sampling(data: pd.DataFrame, n_samples: int, replacement: bool) -> pd.DataFrame: 

    '''
        Performs a Simple Random Sampling operation. 

        Parameters: 
        - data (pd.DataFrame): pandas dataframe containing all the data. 
        - n_samples (int): number of samples the use wishes to extract. 
        - replacement (bool): Tells the program if samples will be put back into the population or not.

        Returns: pd.Dataframe of randomly sampled rows in the data. 
    
    '''

    max_value: int = len(data) - 1
    
    # n_samples cannot exceed population when removing samples from population pool. 
    # max n_samples with replacement = n(population)
    if not replacement and n_samples > max_value:

        raise ValueError('Number of samples cannot exceed number of instances without replacement')

    if replacement: 

        # Create randomly generated indices; with possible duplicates. 
        sample_indices: list = [rd.randint(0, max_value) for i in range(n_samples)]

    else:
        
        # Create randomly generated indices; without the possibility of duplicates
        sample_indices: list = rd.sample(range(len(data)), n_samples)


    samples: pd.DataFrame = data.iloc[sample_indices].reset_index(drop=True)

    return samples

def stratified_sampling(data: pd.DataFrame, attribute: str, n_samples: int) -> pd.DataFrame:

    """
    Performs stratified sampling on a DataFrame based on a specified attribute. The function is
    designed to only split the categories into mod-remainder = 0 type splits to ensure a uniform 
    sample distribution across all bins. 
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - attribute (str): The column name to stratify on.
    - n_samples (int): The total number of samples to draw.
    
    Returns:
    - pd.DataFrame: A stratified sample DataFrame.

    Improvement: 
    - Bend the uniform distribution a tiny bit to get a stronger strata SSE score. 
    - Ensure that frequency differences don't get out of control. 
        - Threshold: +/- (10% of the mean of each partition frequency?)
    - Does it even matter? 
    """
    
    def check_input() -> bool:

        # Raise error if user enters an invalid attribute:
        if attribute not in data.columns:
        
            raise ValueError(f'Attribute: {attribute} does exist in dataset.')

    def divide_data(attribute_values: np.ndarray, k: int) -> list:

        """
        Divides the attribute values into k equal bins.
        
        Parameters:
        - attribute_values (np.ndarray): The values of the attribute to stratify.
        - k (int): Number of bins.
        
        Returns:
        - list: A list of lists, each containing the indices for a bin.
        """

        # Calculate the size of each bin
        div_size = len(attribute_values) // k

        sub_arrs = []

        for i in range(0, len(attribute_values), div_size):

            sub_arrs.append(list(attribute_values[i : i + div_size]))

        return sub_arrs

    def calc_sse(bin_values: list) -> float:
        """
        Calculates the Sum of Squared Errors (SSE) for a bin.
        
        Parameters:
        - bin_values (list): The values in the bin.
        
        Returns:
        - float: The SSE of the bin.
        """
        bin_array = np.array(bin_values)

        mean = bin_array.mean()

        return np.sum((bin_array - mean) ** 2)

    def calc_data(bins: list, k: int) -> float:
        """
        Calculates the penalty score for a given k.
        
        Parameters:
        - bins (list): A list of bins, each bin is a list of attribute values.
        - k (int): Number of bins.
        
        Returns:
        - float: The penalty score.
        """
        sse_cum = 0

        for bin_values in bins:
            sse_cum += calc_sse(bin_values)
        
        penalty_score = (sse_cum + 1) + (2 ** k)

        return float(penalty_score)
    
    def find_optimal_k(attribute_values: np.ndarray) -> int:
        """
        Finds the optimal k (number of bins) that minimizes the penalty score.
        
        Parameters:
        - attribute_values (np.ndarray): The values to stratify.
        
        Returns:
        - int: The optimal number of bins.
        """

        n = len(attribute_values)
        best_k = {}
        best_part = {}
        
        for k in range(1, n + 1):

            if n % k == 0:

                bins = divide_data(attribute_values, k)
                score = calc_data(bins, k)
                best_k[k] = score
                best_part[k] = bins
        
        # Create DataFrame from best_k
        scores_df = pd.DataFrame(list(best_k.items()), columns=['k', 'score'])
        
        # Find the k with the minimum score
        min_score_row = scores_df.loc[scores_df['score'].idxmin()]
        optimal_k = int(min_score_row['k'])
        
        print(scores_df)
        print(f"Optimal k = {optimal_k} with score = {min_score_row['score']}")
        
        return optimal_k, best_part[optimal_k]

    def perform_sampling(bins: list, n_samples: int) -> list:
        """
        Performs stratified sampling from the bins.
        
        Parameters:
        - bins (list): A list of bins, each bin is a list of indices.
        - n_samples (int): Total number of samples desired.
        
        Returns:
        - list: A list of sampled indices.
        """
        k = len(bins)
        samples_per_bin = [n_samples // k] * k
        remainder = n_samples % k
        
        # Distribute the remainder
        for i in range(remainder):
            samples_per_bin[i] += 1
        
        sampled_indices = []

        for bin_indices, samples in zip(bins, samples_per_bin):

            if samples > len(bin_indices):

                raise ValueError(f"Not enough samples in bin to draw {samples} samples.")
            
            sampled = np.random.choice(bin_indices, size=samples, replace=False)

            sampled_indices.extend(sampled)
        
        return sampled_indices

    # Check Data: 
    check_input()

    # Checking for numerical input. 
    if pd.api.types.is_numeric_dtype(data[attribute]):

        # Extract the attribute values and sort them
        attribute_values = data[attribute].values
        sorted_indices = np.argsort(attribute_values)
        sorted_attribute = attribute_values[sorted_indices]
        
        # Find the optimal k and corresponding bins
        optimal_k, optimal_bins = find_optimal_k(sorted_attribute)
        
        # Map sorted bins back to original DataFrame indices
        n = len(sorted_indices)
        div_size = n // optimal_k
        partitioned_indices = []
        
        for i in range(0, n, div_size):

            partitioned_indices.append(sorted_indices[i : i + div_size].tolist())
        
        # Perform stratified sampling
        sampled_indices = perform_sampling(partitioned_indices, n_samples)
        
        # Retrieve the sampled DataFrame
        sampled_df = data.iloc[sampled_indices].reset_index(drop=True)

        return sampled_df
    
    # Categorical input. 
    else:
        
        # Categorical stratification: use pandas' built-in function
        sampled_df = data.groupby(attribute, group_keys=False).apply
        (
            lambda x: x.sample(n=int(np.round(n_samples * len(x) / len(data))), replace=True)
        )

        return sampled_df.reset_index(drop=True)

def train_test_split(X: pd.Series, Y: pd.Series, test_size: float ,method: str, random_state: int = -1) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    def check_data():

        if test_size < 0 or test_size >= 1.00: 

            raise ValueError(f'Parameter: test_size \nmust be between (0, 1) exclusive. \nYou entered: {test_size}')

        if method not in {'cut', 'stratify', 'random'}:

            raise ValueError(f'Parameter: method\nMust be one of the following: [cut, stratify, random].\nYou entered: {method}')

    '''
    TODO: 
    - How are you dividing the data? Randomly or statified?

    Pameters: 
    - X (pd.Series): Complete population attribute array. 
    - Y (pd.Series): Complete population target array. 
    - test_size (float): Percent of population array to be allocated to testing data. 
    - method (str): Method in which the data will be divided by.
        - method = cut: Cut the bottom test_size % of the dataset. 
            - Vulnerable to uneven class distribution if sorted according to Y. 
            - Possible Solution: Shuffle samples n-times with respect to Y. 
        - method = statify: Stratify test sample with uniform % class distribution as training sample. 
        - method = random: Collect n=(test_size * population_size) random samples from the population arrays.
    - random_state (int): Reproduceable random selection. 
    
    Returns:
    - pd.Series (x4): X_train, y_train, X_test, y_test. 
    '''

   
    check_data()

    if random_state != -1:

        np.random.seed(random_state)

    sample_size: int = len(X)
    n_test_samples: int = int(sample_size * test_size)
    data: pd.DataFrame = pd.DataFrame({'X': X, 'Y': Y})

    if method == 'cut':

        x_train: pd.Series = X.iloc[:n_test_samples]
        y_train: pd.Series = Y.iloc[:n_test_samples]
        x_test: pd.Series = X.iloc[n_test_samples:]
        y_test: pd.Series = Y.iloc[n_test_samples:]

        return x_train, y_train, x_test, y_test
    
    elif method == 'random':   
    
        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        test_indices = indices[:n_test_samples]
        train_indices = indices[n_test_samples:]

        x_test = X.iloc[test_indices]
        y_test = Y.iloc[test_indices]
        x_train = X.iloc[train_indices]
        y_train = Y.iloc[train_indices] 

        return x_train, y_train, x_test, y_test
        
    elif method == 'stratify':

        stratified_test = stratified_sampling(data, 'Y', n_samples=n_test_samples)
        test_indices = stratified_test.index

        train_indices = data.index.difference(test_indices)

        x_test = data.loc[test_indices, 'X']
        y_test = data.loc[test_indices, 'Y']
        x_train = data.loc[train_indices, 'X']
        y_train = data.loc[train_indices, 'Y']

        return x_train, y_train, x_test, y_test

    
