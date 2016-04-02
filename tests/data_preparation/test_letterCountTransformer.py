from unittest import TestCase
from data_preparation.transformers import LetterCountTransformer
import pandas as pd



class TestLetterCountTransformer(TestCase):
    pass

class BaseLetterCountTransformerTestCase(TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({'col1': ['a', 'bc', 'def'], 'col2': ['ghi', 'j', 'kl'],'col3': ['mnop', 'qrs', 'tuvwxyz']})

    def tearDown(self):
        self.test_df = None


class TestLetterCountTransformer(BaseLetterCountTransformerTestCase):
    def test_fit_transform_one_column(self):
        expected = pd.DataFrame({0: {0: 1, 1: 2, 2: 3}})
        self.assertTrue(expected.equals(LetterCountTransformer(['col1']).fit_transform(self.test_df)))

    def test_fit_transform_multiple_columns(self):
        expected = pd.DataFrame({0: {0: 1, 1: 2, 2: 3}, 1: {0: 3, 1: 1, 2: 2}, 2: {0: 4, 1: 3, 2: 7}})

        self.assertTrue(expected.equals(LetterCountTransformer(['col1', 'col2', 'col3']).fit_transform(self.test_df)))

