from unittest import TestCase
from kaggle import settings

class BaseSettingsTestCase(TestCase):
    def setUp(self):
        self.settings = settings.Settings()

    def tearDown(self):
        self.settings = None


class TestSettings(BaseSettingsTestCase):

    def test_test_file_paths(self):
        self.assertEqual(self.settings.train_file_path,'..\\data\\train.csv')

    def test_train_file_paths(self):
        self.assertEqual(self.settings.test_file_path, '..\\data\\test.csv')
