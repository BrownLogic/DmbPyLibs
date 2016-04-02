from unittest import TestCase
from data_preparation.transformers import LetterExtractionTransformer
import pandas as pd

class BaseLetterExtractionTransformerTestCase(TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({'col1': ['a', 'bc', 'def'], 'col2': ['ghi', 'j', 'kl'],'col3': ['mnop', 'qrs', 'tuvwxyz']})

    def tearDown(self):
        self.test_df = None


class TestLetterExtractionTransformer(BaseLetterExtractionTransformerTestCase):
    def test_fit_transform_one_column(self):
        expected = pd.DataFrame({0: {0: 'a', 1: 'b', 2: 'd'}, 1: {0: None, 1: 'c', 2: 'e'}, 2: {0: None, 1: None, 2: 'f'}})
        self.assertTrue(expected.equals(LetterExtractionTransformer(['col1']).fit_transform(self.test_df)))

    def test_fit_transform_multiple_columns(self):
        expected = pd.DataFrame({0: {0: 'a', 1: 'b', 2: 'd'}, 1: {0: None, 1: 'c', 2: 'e'}, 2: {0: None, 1: None, 2: 'f'}, 3: {0: 'g', 1: 'j', 2: 'k'}, 4: {0: 'h', 1: None, 2: 'l'}, 5: {0: 'i', 1: None, 2: None}, 6: {0: 'm', 1: 'q', 2: 't'}, 7: {0: 'n', 1: 'r', 2: 'u'}, 8: {0: 'o', 1: 's', 2: 'v'}, 9: {0: 'p', 1: None, 2: 'w'}, 10: {0: None, 1: None, 2: 'x'}, 11: {0: None, 1: None, 2: 'y'}, 12: {0: None, 1: None, 2: 'z'}})

        self.assertTrue(expected.equals(LetterExtractionTransformer(['col1', 'col2', 'col3']).fit_transform(self.test_df)))



