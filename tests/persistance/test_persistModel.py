from unittest import TestCase
from persistance.execution_context import PersistModel
from persistance.execution_context import FileObjectType


class BasePersistModelTestCase(TestCase):
    def setUp(self):
        self.file_loc = r'..\loc_to_write_read_files' #os sensitive
        pass

    def tearDown(self):
        pass


class TestSettings(BasePersistModelTestCase):
    def test_initialization_no_parms(self):
        model_persistor = PersistModel()
        self.assertGreater(model_persistor.run_id, 1)

class TestSavingAndLoadingClasses(BasePersistModelTestCase):
    def test_save_single_object(self):
        model_persistor = PersistModel(object_file_location=self.file_loc )
        run_id = model_persistor.run_id
        something_to_save = {0: {0: 'a', 1: 'b', 2: 'd'},
                             1: {0: 'None', 1: 'c', 2: 'e'},
                             2: {0: 'None', 1: 'None', 2: 'f'},
                             3: {0: 'g', 1: 'j', 2: 'k'},
                             4: {0: 'h', 1: 'None', 2: 'l'},
                             5: {0: 'i', 1: 'None', 2: 'None'},
                             6: {0: 'm', 1: 'q', 2: 't'},
                             7: {0: 'n', 1: 'r', 2: 'u'},
                             8: {0: 'o', 1: 's', 2: 'v'},
                             9: {0: 'p', 1: 'None', 2: 'w'},
                             10: {0: 'None', 1: 'None', 2: 'x'},
                             11: {0: 'None', 1: 'None', 2: 'y'},
                             12: {0: 'None', 1: 'None', 2: 'z'}}

        model_persistor.add_object_to_save(something_to_save, FileObjectType.train_feature)
        model_persistor.save_all_objects()

        model_getter = PersistModel(object_file_location=self.file_loc, run_id = run_id)

        loaded_file = model_getter.get_object(FileObjectType.train_feature)

        self.assertEqual(something_to_save, loaded_file, 'Loaded file does not match saved file')

    def test_save_multiple_objects(self):
            model_persistor = PersistModel(object_file_location=self.file_loc)
            run_id = model_persistor.run_id
            something_to_save = {0: {0: 'a', 1: 'b', 2: 'd'},
                                 1: {0: 'None', 1: 'c', 2: 'e'},
                                 2: {0: 'None', 1: 'None', 2: 'f'},
                                 3: {0: 'g', 1: 'j', 2: 'k'},
                                 4: {0: 'h', 1: 'None', 2: 'l'},
                                 5: {0: 'i', 1: 'None', 2: 'None'},
                                 6: {0: 'm', 1: 'q', 2: 't'},
                                 7: {0: 'n', 1: 'r', 2: 'u'},
                                 8: {0: 'o', 1: 's', 2: 'v'},
                                 9: {0: 'p', 1: 'None', 2: 'w'},
                                 10: {0: 'None', 1: 'None', 2: 'x'},
                                 11: {0: 'None', 1: 'None', 2: 'y'},
                                 12: {0: 'None', 1: 'None', 2: 'z'}}

            model_persistor.add_note("Note 1")

            something_else_to_save = {1: {0: 'a', 1: 'b', 2: 'd'},
                                 1: {1: 'None', 1: 'c', 2: 'e'},
                                 2: {1: 'None', 1: 'None', 2: 'f'},
                                 3: {1: 'g', 1: 'j', 2: 'k'},
                                 4: {1: 'h', 1: 'None', 2: 'l'},
                                 5: {1: 'i', 1: 'None', 2: 'None'},
                                 6: {1: 'm', 1: 'q', 2: 't'},
                                 7: {1: 'n', 1: 'r', 2: 'u'},
                                 8: {1: 'o', 1: 's', 2: 'v'},
                                 9: {1: 'p', 1: 'None', 2: 'w'},
                                 10: {1: 'None', 1: 'None', 2: 'x'},
                                 11: {1: 'None', 1: 'None', 2: 'y'},
                                 12: {1: 'None', 1: 'None', 2: 'z'}}

            model_persistor.add_note("Note 2")

            model_persistor.add_object_to_save(something_to_save, FileObjectType.train_feature)
            model_persistor.add_object_to_save(something_else_to_save, FileObjectType.predictor_model)

            model_persistor.save_all()

            model_getter = PersistModel(object_file_location=self.file_loc, run_id=run_id)

            # Test the loaded files
            loaded_file = model_getter.get_object(FileObjectType.train_feature)

            self.assertEqual(something_to_save, loaded_file, 'Loaded file does not match saved file')

            another_loaded_file = model_getter.get_object(FileObjectType.predictor_model)
            self.assertEqual(something_else_to_save, another_loaded_file, 'Another Loaded file does not match saved file')

            # Test the messages
            # The expectation is that the messages are identical
            self.assertListEqual(model_persistor.notes, model_getter.notes)









