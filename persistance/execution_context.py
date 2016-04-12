"""
Excecution_context.py:  Used for establishing a record of run execution.
Currently it's backed by a database. I may abstract it later for other types of back ends.
"""
import MySQLdb
import cPickle
import gzip
from database import login_info
from datetime import datetime
import os
from data_preparation import transformers
import numpy as np

_FILE_NAME_TEMPLATE = "R{run_id:06d}_{time_stamp:%Y%m%d_%H%M%S}_{type}_{obj_num:02d}"
_RUN_TIME_STAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
_NOTE_TIME_STAMP_FORMAT = '%Y-%m-%d %H:%M:%S.%f' # Includes microseconds
_MAX_ALLOWED_PACKET_SIZE = 500 * 1024 * 1024 #500MB

class FileObjectType():
    """
    A static class for providing consistent name conventions for classes we are saving.
    The values are used for naming files, so keep that in mind as you add to it.
    """
    train_feature = 'train_feature'
    train_target = 'train_target'
    feature_model = 'feature_model'
    predictor_model = 'predictor_model'
    grid_search = 'grid_search'
    submission_file = 'submission'

    def __init__(self):
        pass

class ObjectInfo():
    """
    This class holds objects and context.  Used as a staging object before saving and loading data
    """
    def __init__(self, obj_type=None, obj=None, obj_info=None, file_name=None ):
        self.obj_type = obj_type
        self.obj = obj
        self.obj_info = obj_info
        self.file_name = file_name

class PersistModel:
    """
    This class contains methods for organizing and storing context for specific models and runs.
    It provides methods for initializing, storing and retrieving context to database and instantiated objects to file.
    When instantiated without a run_id, a new one is created.
    When instantiated with a run_id, the context is loaded from the database
    Files and previously instantiated objects are loaded the first time they are requested.
    """
    def __init__(self, object_file_location, project_name, connection=None,  run_id=None, start_time=None):
        """
        Initializes class
        :param connection: Open MySQLdb connection
        :param object_file_location: the directory where object files will be (have been) stored
        :param run_id:  Run ID to initialize.  If not passed a new one is created
        :param start_time: DateTime associated as the start time.  If not passed, one is started initially
        :return:
        """
        self._db = connection or self._create_connection()
        self._db.set_character_set('utf8')
        self.notes = []
        self._objects_to_save = []
        self._object_file_location = object_file_location
        self.start_time = start_time or datetime.now()
        self.scores = []
        self.project_name = project_name
        self.grid_scores = []

        if run_id:
            # load information from database
            self.run_id = run_id
            self._load_from_database()

        else:
            # start new instance
            self.start_time = start_time or datetime.now()  # used for time_stamp and time stamp
            self.run_id = self.start_model()


    def save_all(self):
        self.save_all_objects()
        self.save_all_run_notes()
        self.save_all_scores()
        self._save_all_grid_scores()

    def add_note(self, note, time_stamp=datetime.now()):
        """
        Add's a note to be saved and includes a timestamp
        """
        self.notes.append((time_stamp, note))

    def add_score(self, score_type, score):
        self.scores.append((score_type, score))

    def add_grid_scores(self, grid_scores):
        self.grid_scores = grid_scores

    # todo:  consider what needs to happen on a 'resave'.  Version or replace?  Maybe only save 'new'
    def add_object_to_save(self, the_object, obj_type):
        """
        Accepts a new item and adds it to the list. Objects aren't saved until save is called.
        """

        self._objects_to_save.append(ObjectInfo(obj_type=obj_type, obj=the_object, obj_info=str(the_object)))


    # todo:  I need a better way to load objects.  Expecially if there are multiples of the same type.
    def get_object(self, object_type):
        """
        Returns a instance of an object in the list
        if the object hasn't been loaded, it will be
        loaded from disk
        """
        ret = filter(lambda x: x.obj_type == object_type, self._objects_to_save)
        if len(ret) == 0:
            raise Exception('No objects of type {} found'.format(object_type))

        if len(ret) > 1:
            raise Exception('Multiple types of object {} found'.format(object_type))

        if ret[0].obj is None:
            ret[0].obj = self._load_zipped_pickle(self._get_file_path(ret[0].file_name))

        return ret[0].obj

    def _load_from_database(self):
        """
        Get's data from database.
        """

        sql = ' SELECT project_name, run_time_stamp ' \
              ' FROM model_runs ' \
              ' WHERE run_id = {}'.format(self.run_id)

        cursor = self._db.cursor()
        cursor.execute(sql)
        model_info = cursor.fetchall()
        cursor.close()
        #Expect only one record but might not get any
        if len(model_info) <> 1:
            raise Exception('Run_id {} not found or found too many times'.format(self.run_id))
        self.project_name, self.start_time = model_info[0]
        self._load_notes_from_database()
        self._load_object_info_from_database()
        self._load_scores_from_database()


    def _load_all_objects(self):
        """
        Helper function that loads all of the objects from file
        The assumption is that the instance was loaded by run_id (rather than from scratch)
        """
        for obj_info in self._objects_to_save:
            obj_info.obj = self._load_zipped_pickle(self._get_file_path(obj_info.file_name))

    def _load_notes_from_database(self):
        """
        Get's the notes data from database.
        """

        sql = ' SELECT note_time_stamp, sequence_num, note ' \
              ' FROM model_run_notes ' \
              ' WHERE run_id = {} ' \
              ' ORDER BY sequence_num'.format(self.run_id)

        cursor = self._db.cursor()
        cursor.execute(sql)
        note_info = cursor.fetchall()
        cursor.close()
        # Expect only one record but might not get any
        for (note_time_stamp, sequence_num, note) in note_info:
            self.add_note(note, note_time_stamp)

    def _load_scores_from_database(self):
        """
        Get's the score data from database.
        """

        sql = ' SELECT score_type, score' \
              ' FROM model_scores ' \
              ' WHERE run_id = {} '.format(self.run_id)

        cursor = self._db.cursor()
        cursor.execute(sql)
        score_info = cursor.fetchall()
        cursor.close()
        # Expect only one record but might not get any
        for (score_type, score) in score_info:
            self.add_score(score_type=score_type, score=score)

    def _load_object_info_from_database(self):
        """
        Get's the notes data from database.
        """

        sql = ' SELECT sequence_num, obj_type, object_info, file_name ' \
              ' FROM model_run_object_info ' \
              ' WHERE run_id = {} ' \
              ' ORDER BY sequence_num'.format(self.run_id)

        cursor = self._db.cursor()
        cursor.execute(sql)
        object_info = cursor.fetchall()
        cursor.close()
        # Expect only one record but might not get any
        for (sequence_num, obj_type, object_info, file_name) in object_info:
            self._objects_to_save.append(ObjectInfo(obj_type=obj_type, obj_info=object_info, file_name=file_name))

    def start_model(self):
        """
        Adds a record to the database and returns the run id
        :return row_id: the row_id of the new model run
        """

        sql = " INSERT INTO model_runs " \
              " (project_name, run_time_stamp) " \
              " VALUES (%s, %s)"
        cursor = self._db.cursor()

        cursor.execute(sql, [self.project_name, self.run_time_stamp() ])
        row_id = cursor.lastrowid
        cursor.close()
        self._db.commit()
        return row_id


    def save_submission(self, data, location):
        """
        Saves the submission file to the target location following naming pattern.
        :param data DataFrame of submission data
        :param location:  directory for submission file
        :return file_name:  name of the file created
        """
        file_name = self._get_file_name(FileObjectType.submission_file, 0)+'.csv'
        data.to_csv(path_or_buf=os.path.join(location, file_name), header=True, index=False, encoding='utf-8')
        return file_name


    def run_time_stamp(self):
        """
        returns the timestamp as string in appropriate format
        """
        return datetime.strftime(self.start_time,_RUN_TIME_STAMP_FORMAT)

    def get_log_context(self):
        """
        Returns a context snapshot including run_id and time_stamp
        Handy for logs and messages
        """
        return " run {}, timestamp {}".format(self.run_id, self.run_time_stamp())


    def save_all_objects(self):
        """
        Saves all of the objects in the object collection.  Saves context to database and files
        to the folder location specified at instantiation.
        """
        for sequence_num, obj_info  in enumerate(self._objects_to_save):
            obj_info.file_name = self._get_file_path(self._get_file_name(obj_info.obj_type, sequence_num))
            self._save_object(sequence_num, obj_info.obj_type, obj_info.obj,obj_info.file_name)

    def _save_object(self, sequence_num, obj_type, obj, file_name):
        """
        Saves the object file and records context to the database
        """
        file_name = self._get_file_name(obj_type, sequence_num)
        self._save_zipped_pickle(obj, self._get_file_path(file_name))
        self._save_object_context(sequence_num, obj_type, obj, file_name)

    def _get_string_representation_object_info(self, obj_type, obj):
        if obj_type in [FileObjectType.predictor_model, FileObjectType.feature_model]:
            ret = transformers.encode_transformer(obj_type, obj)
        else:
            ret = str(type(obj))

        return ret

    def _save_object_context(self, sequence_num, obj_type, obj, file_name):
        """
        Adds a record to the database and returns the model_run_object_id
        """
        sql = " INSERT INTO model_run_object_info " \
              " ( run_id, sequence_num, obj_type, object_info, file_name ) " \
              " VALUES (%s, %s, %s, %s, %s)"

        cursor = self._db.cursor()
        cursor.execute(sql, [self.run_id, sequence_num, obj_type, self._get_string_representation_object_info(obj_type, obj), file_name])
        model_run_object_id = cursor.lastrowid
        cursor.close()
        self._db.commit()
        return model_run_object_id

    def save_all_scores(self):
        """
        Saves all of the notes in the collection to database.
        """
        for score_type, score in self.scores:
            self._save_score(score_type, score)

    def save_all_run_notes(self):
        """
        Saves all of the notes in the collection to database.
        """
        for idx, (timestamp, note) in enumerate(self.notes):
            self._save_run_note(idx, timestamp, note)


    def _save_score(self, score_type, score):
        """
        Adds a record to the database and returns the score_id
        """
        sql = " INSERT INTO model_scores " \
              " (run_id, score_type, score ) " \
              " VALUES (%s, %s, %s)"
        cursor = self._db.cursor()
        cursor.execute(sql, [self.run_id, score_type, round(score,8)])
        score_id = cursor.lastrowid
        cursor.close()
        self._db.commit()
        return score_id

    def _save_run_note(self, sequence_num, timestamp, note):
        """
        Adds a record to the database and returns the model_run_object_id
        """
        sql = " INSERT INTO model_run_notes " \
              " (run_id, sequence_num, note_time_stamp, note ) " \
              " VALUES (%s, %s, %s, %s)"
        cursor = self._db.cursor()
        cursor.execute(sql, [self.run_id, sequence_num, datetime.strftime(timestamp,_NOTE_TIME_STAMP_FORMAT), note])
        run_note_id = cursor.lastrowid
        cursor.close()
        self._db.commit()
        return run_note_id

    def _get_file_path(self, file_name):
        """
        Returns the path and directory for the file_name
        """
        return os.path.join(self._object_file_location, file_name)

    def _save_grid_score(self, mean, std, params):
        """
        This function saves one grid score record to the database
        :param mean: the mean_validation_score
        :param std:  standard deviation of cv_validation_scores
        :param params:  the parameters that resulted in the score
        """
        sql = " INSERT INTO grid_search_results " \
              " (run_id, mean, std, params) " \
              " VALUES (%s, %s, %s, %s) "
        cursor = self._db.cursor()
        cursor.execute(sql, [self.run_id, round(mean, 8), round(std, 8), params])
        self._db.commit()


    def _save_all_grid_scores(self):
        for score in self.grid_scores:
            self._save_grid_score(score.mean_validation_score,
                                 np.std(score.cv_validation_scores),
                                 str(score.parameters))

    def _get_file_name(self, object_type, obj_num):
        """
        Formats the file name according to the default template
        """
        return _FILE_NAME_TEMPLATE.format(**{'type': object_type, 'run_id': self.run_id,
                                             'time_stamp': self.start_time, 'obj_num': obj_num}
                                          )

    def _save_zipped_pickle(self, obj, file_path_name, protocol=-1):
        """
        From http://stackoverflow.com/questions/18474791/decreasing-the-size-of-cpickle-objects
        Saves a file both pickled and zipped
        """
        with gzip.open(file_path_name, 'wb') as f:
            cPickle.dump(obj, f, protocol)

    def _load_zipped_pickle(self, file_path_name):
        """
        From http://stackoverflow.com/questions/18474791/decreasing-the-size-of-cpickle-objects
        loads a file that has been both pickled and zipped
        """
        with gzip.open(file_path_name, 'rb') as f:
            loaded_object = cPickle.load(f)
            return loaded_object

    def _create_connection(self):
        """
        creates a MySQLdb.connection
        """
        ret = MySQLdb.connect(**login_info)
        """
        Some of the information in this get's big.  I need to check the value of max_global packet.  If it's already bigger, leave it.  If it's smaller, update
        """
        sql = 'select @@global.max_allowed_packet'

        cursor = ret.cursor()
        cursor.execute(sql)
        max_allowed_packet_value = cursor.fetchone()
        if max_allowed_packet_value < _MAX_ALLOWED_PACKET_SIZE:
            sql = 'SET @@global.max_allowed_packet = {}'.format(_MAX_ALLOWED_PACKET_SIZE)
            cursor.execute(sql)

        return ret



    def _get_start_date_from_run_time_stamp(self, run_time_stamp):
        """
        Converts a run_time_stamp (assumed to be the appropriate format) to a
        date time object
        :return DateTime instantiation of timestamp.
        """
        return datetime.strptime(run_time_stamp, _RUN_TIME_STAMP_FORMAT)