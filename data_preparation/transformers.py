"""
transformers.py:  a library of transformer and transformer like classes that can be used in pipelines, grid search, etc.
"""

import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder


class NaNCountTransformer(TransformerMixin):
    """
    This counts the number of NaN per row and provides a raw count
    and percentage
    """

    def transform(self, X, **transform_params):
        nan_metrics = pd.DataFrame(len(X.columns)-X.count(axis=1))
        nan_metrics = pd.concat([nan_metrics, pd.DataFrame(nan_metrics[0]/len(X.columns))], axis=1)
        return nan_metrics

    def fit(self, X, y=None, **fit_params):
        return self


class NanToZeroTransformer(TransformerMixin):
    """
    Accepts an array of values and converts NaN to 0
    """
    def transform(self, X, **transform_params):
        return np.nan_to_num(X)

    def fit(self, X, y=None, **fit_params):
        return self


class ColumnExtractor(TransformerMixin):
    """
    Slices a dataframe and returns specified columns
    (mirrored from Zac Stewart)

        :param cols_to_return: list of columns
    """

    def __init__(self, cols_to_return):
        self.cols = cols_to_return

    def transform(self, X, **transform_params):
        return pd.DataFrame(X[self.cols])

    def fit(self, X, y=None, **fit_params):
        return self


class LetterExtractionTransformer(TransformerMixin):
    """
    Accepts a column name and creates an m x n array where m is number of rows
    and n is max(length of column) populated by letters (or empty where shorter)

    @:param columns_to_extract: the columns to be 'split'
    """
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, **transform_params):
        return pd.concat([pd.DataFrame(map(self.my_list, X[col]))
                          for col in self.columns], axis=1, ignore_index=True)


    def fit(self, X, y=None, **fit_params):
        return self

    def my_list(self, x):
        if type(x) == str:
            return list(x)
        return []

class MultiColumnLabelEncoder(TransformerMixin):
    """
    Accepts a list of columns and label encodes them.
    Copied from http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    @param columns: array of columns to encode
    """
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output


class LetterCountTransformer(TransformerMixin):
    """
    Gets the count of letters
    @ param columns: list of columns to get counts for
    """

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def transform(self, X, **transform_params):
        return pd.concat([pd.DataFrame(map(self.get_len, X[col])) for col in self.columns], axis=1, ignore_index=True)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_len(self, some_string):
        if isinstance(some_string, basestring):
            return len(some_string)
        return np.NaN