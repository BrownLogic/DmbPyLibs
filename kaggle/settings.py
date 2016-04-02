import os, json

class Settings():
    def __init__(self):
        with open(os.path.join(os.pardir, 'SETTINGS.json')) as f:
            self.raw_json = json.load(f)
        self.train_file_path = os.path.join(os.path.pardir, self.raw_json['data_directory_name'], self.raw_json['train_file'] )
        self.test_file_path = os.path.join(os.path.pardir, self.raw_json['data_directory_name'], self.raw_json['test_file'])
        self.saved_object_directory = os.path.join(os.path.curdir, self.raw_json['saved_object_directory_name'])
        self.submissions_directory =  os.path.join(os.path.curdir, self.raw_json['submission_directory_name'])
        self.non_features = self.raw_json['data_info']['non_features']
        self.string_features = self.raw_json['data_info']['string_features']
        self.special_string_features = self.raw_json['data_info']['special_string_features']
        self.numeric_features = self.raw_json['data_info']['numeric_features']
        self.logging_config = os.path.join(os.path.pardir, self.raw_json['logging_config_file'] )
        self.all_features = self.string_features+self.numeric_features+self.special_string_features
        self.target = self.raw_json['data_info']['target']
        self.record_pk = self.raw_json['data_info']['record_pk']

