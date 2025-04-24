from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
import warnings
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
from sklearn.impute import KNNImputer
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        import warnings
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
      

class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one-hot encoding on a specified target column.

    This transformer uses pandas' get_dummies function to create one-hot encoded
    columns from a categorical column. The resulting columns are prefixed with
    the target column name.

    Parameters
    ----------
    target_column : str
        The name of the column to be one-hot encoded.

    Attributes
    ----------
    target_column : str
        The name of the column to be one-hot encoded.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'joined': ['Belfast', 'Cherbourg', 'Queenstown']})
    >>> ohe = CustomOHETransformer(target_column='joined')
    >>> transformed_df = ohe.fit_transform(df)
    >>> transformed_df.columns.tolist()
    ['joined_Belfast', 'joined_Cherbourg', 'joined_Queenstown']
    """

    def __init__(self, target_column: str) -> None:
        """
        Initialize the CustomOHETransformer.

        Parameters
        ----------
        target_column : str
            The name of the column to be one-hot encoded.
        """
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomOHETransformer
            Returns self to allow method chaining.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by one-hot encoding the target column.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to be one-hot encoded.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the target column replaced by one-hot encoded columns.

        Raises
        ------
        AssertionError
            If the target column doesn't exist in X.
        """
        # Verify that the target column exists in the DataFrame
        assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column \"{self.target_column}\""

        # Perform one-hot encoding using pd.get_dummies
        dummies = pd.get_dummies(X[self.target_column], prefix=self.target_column)

        # Convert boolean values to integers (0 and 1)
        dummies = dummies.astype(int)

        # Create a copy of the original DataFrame
        X_transformed = X.copy()

        # Drop the original target column
        X_transformed = X_transformed.drop(columns=[self.target_column])

        # Add the one-hot encoded columns to the DataFrame
        for col in dummies.columns:
            X_transformed[col] = dummies[col]

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to be one-hot encoded.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the target column one-hot encoded.
        """
        return self.transform(X)


class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomDropColumnsTransformer
            Returns self to allow method chaining.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by either dropping or keeping specified columns.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with columns dropped or kept as specified.

        Raises
        ------
        AssertionError
            If any columns to keep do not exist in the DataFrame (when action='keep').
        Warns
        -----
        Warning
            If any columns to drop do not exist in the DataFrame (when action='drop').
        """
        # Ensure X is a DataFrame
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead."

        # Create a copy of the DataFrame
        X_transformed = X.copy()

        # Get the set of columns in the DataFrame
        df_columns = set(X.columns)

        # Get the set of columns specified in column_list
        specified_columns = set(self.column_list)

        # Find columns that are in the column_list but not in the DataFrame
        missing_columns = specified_columns - df_columns

        if self.action == 'keep':
            # For 'keep' action, it's an error if specified columns are missing
            if missing_columns:
                missing_str = ", ".join([f"'{col}'" for col in missing_columns])
                assert not missing_columns, f"Columns {{{missing_str}}}, are not in the data table"

            # Keep only the specified columns
            columns_to_keep = [col for col in self.column_list if col in df_columns]
            X_transformed = X_transformed[columns_to_keep]

        elif self.action == 'drop':
            # For 'drop' action, just issue a warning if specified columns are missing
            if missing_columns:
                missing_str = ", ".join([f"'{col}'" for col in missing_columns])
                print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {missing_str}.\n")

            # Drop the specified columns that exist in the DataFrame
            columns_to_drop = [col for col in self.column_list if col in df_columns]
            X_transformed = X_transformed.drop(columns=columns_to_drop)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with columns dropped or kept as specified.
        """
        return self.transform(X)


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    
    def __init__(self, target_column):
        self.target_column = target_column
        self.high_wall = None
        self.low_wall = None
    
    def fit(self, X, y=None):
        """
        Compute the 3-sigma boundaries for the target column.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : returns an instance of self.
        """
        assert isinstance(X, pd.DataFrame), f'expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected numeric column but got {X[self.target_column].dtype}'
        
        # Compute mean and standard deviation
        mean = X[self.target_column].mean()
        std = X[self.target_column].std()
        
        # Compute boundaries
        self.low_wall = mean - 3 * std
        self.high_wall = mean + 3 * std
        
        return self
    
    def transform(self, X):
        """
        Apply 3-sigma clipping to the target column.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with the target column clipped according to 3-sigma rule.
        """
        assert self.high_wall is not None and self.low_wall is not None, "Sigma3Transformer.fit has not been called yet."
        assert isinstance(X, pd.DataFrame), f'expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        
        # Create a copy of the input DataFrame
        X_transformed = X.copy()
        
        # Apply clipping
        X_transformed[self.target_column] = X_transformed[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit the transformer and then transform the data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with the target column clipped according to 3-sigma rule.
        """
        return self.fit(X).transform(X)


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.
    
    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.
    
    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).
    
    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    
    def __init__(self, target_column, fence='outer'):
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.outer_low = None
        self.inner_high = None
        self.outer_high = None
    
    def fit(self, X, y=None):
        """
        Compute Tukey's fences for the target column.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : returns an instance of self.
        """
        assert isinstance(X, pd.DataFrame), f'expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected numeric column but got {X[self.target_column].dtype}'
        assert self.fence in ['inner', 'outer'], f'fence must be either "inner" or "outer", got {self.fence}'
        
        # Compute quartiles and IQR
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1
        
        # Compute inner and outer fences
        self.inner_low = q1 - 1.5 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_low = q1 - 3.0 * iqr
        self.outer_high = q3 + 3.0 * iqr
        
        return self
    
    def transform(self, X):
        """
        Apply Tukey's fences to the target column.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with the target column clipped according to Tukey's fences.
        """
        assert self.inner_low is not None and self.inner_high is not None, "TukeyTransformer.fit has not been called yet."
        assert isinstance(X, pd.DataFrame), f'expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        
        # Create a copy of the input DataFrame
        X_transformed = X.copy()
        
        # Apply clipping based on fence type
        if self.fence == 'inner':
            X_transformed[self.target_column] = X_transformed[self.target_column].clip(lower=self.inner_low, upper=self.inner_high)
        else:  # outer fence
            X_transformed[self.target_column] = X_transformed[self.target_column].clip(lower=self.outer_low, upper=self.outer_high)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit the transformer and then transform the data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the target column.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with the target column clipped according to Tukey's fences.
        """
        return self.fit(X).transform(X)

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
    """
    
    def __init__(self, column):
        """Initialize the transformer with the target column name."""
        self.target_column = column
        self.iqr = None
        self.med = None
    
    def fit(self, X, y=None):
        """Compute the IQR and median of the target column.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Verify that the target column exists in the DataFrame
        if self.target_column not in X.columns:
            raise AssertionError(f"CustomRobustTransformer.fit unrecognizable column {self.target_column}.")
        
        # Calculate the 25th and 75th percentiles
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        
        # Calculate IQR
        self.iqr = q3 - q1
        
        # Calculate median
        self.med = X[self.target_column].median()
        
        return self
    
    def transform(self, X):
        """Scale the target column using the IQR and median.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The input data to transform.
            
        Returns
        -------
        X_transformed : pandas.DataFrame
            The transformed DataFrame with the target column scaled.
        """
        # Check if the transformer is fitted
        if self.iqr is None or self.med is None:
            raise AssertionError("NotFittedError: This CustomRobustTransformer instance is not fitted yet.")
        
        # Make a copy of the input DataFrame to avoid modifying the original
        X_transformed = X.copy()
        
        # Apply the transformation only if IQR is not zero (to handle binary columns)
        if self.iqr != 0:
            X_transformed[self.target_column] = (X[self.target_column] - self.med) / self.iqr
        
        return X_transformed

class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """Imputes missing values using KNN.

    This transformer wraps the KNNImputer from scikit-learn and hard-codes
    add_indicator to be False. It also ensures that the input and output
    are pandas DataFrames.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. Possible values:
        "uniform" : uniform weights. All points in each neighborhood
        are weighted equally.
        "distance" : weight points by the inverse of their distance.
        in this case, closer neighbors of a query point will have a
        greater influence than neighbors which are further away.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            add_indicator=False  # Hard-coded to False as required
        )
        self.fitted = False  # Keep track of whether the transformer has been fitted
        
    def fit(self, X, y=None):
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : pandas DataFrame
            Input data, where rows are samples and columns are features.
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Fit the KNNImputer
        self.imputer.fit(X)
        self.fitted = True  # Mark as fitted
        self.columns_ = X.columns  # Store column names for transform
        return self
    
    def transform(self, X):
        """Impute missing values in X.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input data to complete.
            
        Returns
        -------
        pandas DataFrame
            The imputed dataframe.
        """
        # Check if fitted
        if not self.fitted:
            raise ValueError("This CustomKNNTransformer instance is not fitted yet. "
                             "Call 'fit' before calling 'transform'.")
        
        # Transform using the KNNImputer
        imputed_array = self.imputer.transform(X)
        
        # Return as pandas DataFrame with original column names
        return pd.DataFrame(imputed_array, index=X.index, columns=X.columns)
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : pandas DataFrame
            Input data, where rows are samples and columns are features.
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        pandas DataFrame
            The imputed dataframe.
        """
        return self.fit(X).transform(X)


# Pipelines
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),    #from last chapter
    ('scale_fare', CustomRobustTransformer('Fare')),  #from last chapter
], verbose=True)

transformed_df = titanic_transformer.fit_transform(titanic_features)


#Build pipeline and include scalers from last chapter and imputer from this
customer_transformer = Pipeline(steps=[
    ('dropper', CustomDropColumnsTransformer(column_list=[ 'First timer', 'Rating'], action='drop')),
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('experience', CustomMappingTransformer('Experience Level', {'low': 0.0, 'medium': 1.0, 'high': 2.0})),
    ('one_os', CustomOHETransformer(target_column='OS')),
    #('os_mapping', CustomMappingTransformer('OS', {'Android': 0.0, 'iOS': 1.0})),
    ('one_isp', CustomOHETransformer(target_column='ISP')),
    #('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ('robust_scaler_time', CustomRobustTransformer('Time Spent')),
    ('robust_scaler_age', CustomRobustTransformer('Age')),
    ('knn_imputer', CustomKNNTransformer(n_neighbors=5, weights='uniform'))
])
