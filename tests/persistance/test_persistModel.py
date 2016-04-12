from unittest import TestCase
from persistance.execution_context import PersistModel
from persistance.execution_context import FileObjectType
from numpy import array

# For Grid Search
import cPickle as pickle

class Dummy():
    """
    A dummy class for saving and reloading.
    """
    def __init__(self, some_dict):
        self.some_dict = some_dict



class BasePersistModelTestCase(TestCase):
    def setUp(self):
        self.file_loc = r'..\loc_to_write_read_files' #os sensitive
        self.project_name = 'unit_test'
        self.grid_search = self._get_grid_search()


    def tearDown(self):
        pass

    def _get_grid_search(self):
        'saved_grid_20160411.pickle'
        with open(self.file_loc + '\saved_grid_20160411.pickle') as f: # also OS sensitive
            grid_search = pickle.load(f)
        return grid_search


class TestSettings(BasePersistModelTestCase):
    def test_initialization_no_parms(self):

        model_persistor = PersistModel(project_name=self.project_name, object_file_location=self.file_loc)
        self.assertGreater(model_persistor.run_id, 1)

class TestSavingAndLoadingClasses(BasePersistModelTestCase):
    def test_save_single_object(self):
        model_persistor = PersistModel(project_name = self.project_name, object_file_location=self.file_loc )
        run_id = model_persistor.run_id

        something_to_save = Dummy({0: {0: 'a', 1: 'b', 2: 'd'},
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
                             12: {0: 'None', 1: 'None', 2: 'z'}})

        model_persistor.add_object_to_save(something_to_save, FileObjectType.train_feature)
        model_persistor.save_all_objects()

        model_getter = PersistModel(project_name=self.project_name, object_file_location=self.file_loc, run_id = run_id)

        loaded_file = model_getter.get_object(FileObjectType.train_feature)

        self.assertEqual(something_to_save, loaded_file, 'Loaded file does not match saved file')

    def test_save_multiple_objects(self):
            model_persistor = PersistModel(project_name=self.project_name, object_file_location=self.file_loc)
            run_id = model_persistor.run_id
            something_to_save = Dummy({0: {0: 'a', 1: 'b', 2: 'd'},
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
                                 12: {0: 'None', 1: 'None', 2: 'z'}})

            model_persistor.add_note("Note 1")

            something_else_to_save = Dummy({1: {0: 'a', 1: 'b', 2: 'd'},
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
                                 12: {1: 'None', 1: 'None', 2: 'z'}})

            model_persistor.add_note("Note 2")

            model_persistor.add_object_to_save(something_to_save, FileObjectType.train_feature)
            model_persistor.add_object_to_save(something_else_to_save, FileObjectType.predictor_model)

            model_persistor.add_score('test', .25)
            self.assertEqual(len(model_persistor.scores),1)
            model_persistor.add_score('test_2', .75)
            self.assertEqual(len(model_persistor.scores), 2)

            model_persistor.save_all()

            model_getter = PersistModel(project_name=self.project_name, object_file_location=self.file_loc, run_id=run_id)

            self.assertEqual(model_persistor.project_name, model_getter.project_name, 'Project names are different')

            # Test the loaded files
            loaded_file = model_getter.get_object(FileObjectType.train_feature)

            self.assertEqual(something_to_save.some_dict, loaded_file.some_dict, 'Loaded file does not match saved file')

            another_loaded_file = model_getter.get_object(FileObjectType.predictor_model)
            self.assertEqual(something_else_to_save.some_dict, another_loaded_file.some_dict, 'Another Loaded file does not match saved file')

            # Test the messages
            # The expectation is that the messages are identical
            self.assertListEqual(model_persistor.notes, model_getter.notes)

            self.assertEqual(model_persistor.scores, model_getter.scores)


    def test_save_grid_search(self):
        model_persistor = PersistModel(project_name=self.project_name, object_file_location=self.file_loc)
        run_id = model_persistor.run_id
        save_as_feature_model = self.grid_search
        model_persistor.add_note("test_save_grid_search Note 1")

        save_as_predictor_model = self.grid_search.best_estimator_
        model_persistor.add_note("test_save_grid_search Note 1")

        model_persistor.add_object_to_save(save_as_feature_model, FileObjectType.feature_model)
        model_persistor.add_object_to_save(save_as_predictor_model, FileObjectType.predictor_model)

        model_persistor.add_score('test', self.grid_search.best_score_)
        self.assertEqual(len(model_persistor.scores), 1)
        model_persistor.add_score('test_2', 1 - self.grid_search.best_score_)
        self.assertEqual(len(model_persistor.scores), 2)

        model_persistor.add_grid_scores(self.grid_search.grid_scores_)

        model_persistor.save_all()

        model_getter = PersistModel(project_name=self.project_name, object_file_location=self.file_loc, run_id=run_id)

        self.assertEqual(model_persistor.project_name, model_getter.project_name, 'Project names are different')

        # Test the messages
        # The expectation is that the messages are identical
        self.assertListEqual(model_persistor.notes, model_getter.notes)

        self.assertEqual(model_persistor.scores, model_getter.scores)

        # TODO: Correct this comparision.  It doesn't work because the scores are distilled (mean and STD)
         # self.assertListEqual(model_persistor.grid_scores, model_getter.grid_scores)








