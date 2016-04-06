from unittest import TestCase, skip
from data_preparation import transformers

from sklearn.pipeline import FeatureUnion, Pipeline
from data_preparation.transformers import ColumnExtractor
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

class BaseTransformerTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestDescribe_transformers(BaseTransformerTestCase):

    def test_single_transformer(self):
        the_transformer = OneHotEncoder(sparse=True)
        result = str(transformers.encode_transformer("single_one_hot_encoder", the_transformer))
        expected_result = "foo"
        """
        {
            "A. label": "single_one_hot_encoder",
            "B. transformer": "<class 'sklearn.preprocessing.data.OneHotEncoder'>",
            "D. vars": {
                "categorical_features": "all",
                "dtype": "<type 'float'>",
                "handle_unknown": "error",
                "n_values": "auto",
                "sparse": true
            }
        }
        """


        print result
        self.assertEqual(expected_result,result, "single transformer doesn't match")

    def test_pipeline(self):
        the_transformer = Pipeline([('pipeline_one_hot_encoder', OneHotEncoder(sparse=True)), ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=3))])
        result = str(transformers.encode_transformer("pipeline", the_transformer))
        print result
        expected_result = "foo"
        """
        {
            "A. label": "pipeline",
            "B. transformer": "<class 'sklearn.pipeline.Pipeline'>",
            "C. children": [
                {
                    "A. label": "pipeline_one_hot_encoder",
                    "B. transformer": "<class 'sklearn.preprocessing.data.OneHotEncoder'>",
                    "D. vars": {
                        "categorical_features": "all",
                        "dtype": "<type 'float'>",
                        "handle_unknown": "error",
                        "n_values": "auto",
                        "sparse": true
                    }
                },
                {
                    "A. label": "random_forest",
                    "B. transformer": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
                    "D. vars": {
                        "base_estimator": "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,\n            min_samples_split=2, min_weight_fraction_leaf=0.0,\n            presort=False, random_state=None, splitter='best')",
                        "bootstrap": true,
                        "class_weight": null,
                        "criterion": "gini",
                        "estimator_params": [
                            "criterion",
                            "max_depth",
                            "min_samples_split",
                            "min_samples_leaf",
                            "min_weight_fraction_leaf",
                            "max_features",
                            "max_leaf_nodes",
                            "random_state"
                        ],
                        "estimators_": [],
                        "max_depth": 3,
                        "max_features": "auto",
                        "max_leaf_nodes": null,
                        "min_samples_leaf": 1,
                        "min_samples_split": 2,
                        "min_weight_fraction_leaf": 0.0,
                        "n_estimators": 100,
                        "n_jobs": 1,
                        "oob_score": false,
                        "random_state": null,
                        "verbose": 0,
                        "warm_start": false
                    }
                }
            ]
        }
        """

        self.assertEqual(expected_result, result, "pipeline transformer doesn't match")

    def test_grid_search(self):
        parameters = {'N_estimators': (10, 100), 'max_depth': (4, 6, 8),
                      'max_leaf_nodes': (3, 6, 9)}
        the_transformer = GridSearchCV(RandomForestClassifier(), param_grid =parameters, n_jobs=-1)
        result = str(transformers.encode_transformer("grid_search", the_transformer))
        print result
        expected_result = "foo"
        """
         {
            "A. label": "grid_search",
            "B. transformer": "<class 'sklearn.grid_search.GridSearchCV'>",
            "D. vars": {
                "cv": null,
                "error_score": "raise",
                "estimator": "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n            max_depth=None, max_features='auto', max_leaf_nodes=None,\n            min_samples_leaf=1, min_samples_split=2,\n            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,\n            oob_score=False, random_state=None, verbose=0,\n            warm_start=False)",
                "fit_params": {},
                "iid": true,
                "n_jobs": -1,
                "param_grid": {
                    "N_estimators": [
                        10,
                        100
                    ],
                    "max_depth": [
                        4,
                        6,
                        8
                    ],
                    "max_leaf_nodes": [
                        3,
                        6,
                        9
                    ]
                },
                "pre_dispatch": "2*n_jobs",
                "refit": true,
                "scoring": null,
                "verbose": 0
            },
            "E. Estimator": {
                "A. label": "grid_search nested_estimator",
                "B. transformer": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
                "D. vars": {
                    "base_estimator": "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,\n            min_samples_split=2, min_weight_fraction_leaf=0.0,\n            presort=False, random_state=None, splitter='best')",
                    "bootstrap": true,
                    "class_weight": null,
                    "criterion": "gini",
                    "estimator_params": [
                        "criterion",
                        "max_depth",
                        "min_samples_split",
                        "min_samples_leaf",
                        "min_weight_fraction_leaf",
                        "max_features",
                        "max_leaf_nodes",
                        "random_state"
                    ],
                    "estimators_": [],
                    "max_depth": null,
                    "max_features": "auto",
                    "max_leaf_nodes": null,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "min_weight_fraction_leaf": 0.0,
                    "n_estimators": 10,
                    "n_jobs": 1,
                    "oob_score": false,
                    "random_state": null,
                    "verbose": 0,
                    "warm_start": false
                }
            }
        }

        """
        self.assertEqual(expected_result, "grid_search transformer doesn't match")


    def test_feature_union(self):
        the_transformer = FeatureUnion(
            [('column_extractor', ColumnExtractor(["v1", "v2"])), ('one_hot_encoder', OneHotEncoder(sparse=True))])
        result = str(transformers.encode_transformer("feature_union", the_transformer))
        print result
        expected_result = "foo"
        """
        {
            "A. label": "feature_union",
            "B. transformer": "<class 'sklearn.pipeline.FeatureUnion'>",
            "C. children": [
                {
                    "A. label": "column_extractor",
                    "B. transformer": "<class 'data_preparation.transformers.ColumnExtractor'>",
                    "D. vars": {
                        "cols": [
                            "v1",
                            "v2"
                        ]
                    }
                },
                {
                    "A. label": "one_hot_encoder",
                    "B. transformer": "<class 'sklearn.preprocessing.data.OneHotEncoder'>",
                    "D. vars": {
                        "categorical_features": "all",
                        "dtype": "<type 'float'>",
                        "handle_unknown": "error",
                        "n_values": "auto",
                        "sparse": true
                    }
                }
            ]
        }
        """
        self.assertEqual(expected_result, result, "feature_union transformer doesn't match")
        pass


    def test_complex(self):
        steps = [('nfu', FeatureUnion([('nce', ColumnExtractor(['v22'])), ('nohe', OneHotEncoder(sparse=True))])), ('rfc', RandomForestClassifier())]
        parameters =dict(rfc__N_estimators=[10, 100],
                         rfc__max_depth=(4,6,8),
                         rfc__max_leaf_nodes=[3, 6, 9],
                         nohe__sparse=[True, False])

        the_transformer = GridSearchCV(Pipeline(steps), param_grid=parameters, verbose=10)
        result = str(transformers.encode_transformer("test_complex", the_transformer))
        print result
        expected_result = "foo"
        """
        {
            "A. label": "test_complex",
            "B. transformer": "<class 'sklearn.grid_search.GridSearchCV'>",
            "D. vars": {
                "cv": null,
                "error_score": "raise",
                "estimator": "Pipeline(steps=[('nfu', FeatureUnion(n_jobs=1,\n       transformer_list=[('nce', <data_preparation.transformers.ColumnExtractor object at 0x000000000B00D588>), ('nohe', OneHotEncoder(categorical_features='all', dtype=<type 'float'>,\n       handle_unknown='error', n_values='auto', sparse=True))],\n       transf...n_jobs=1,\n            oob_score=False, random_state=None, verbose=0,\n            warm_start=False))])",
                "fit_params": {},
                "iid": true,
                "n_jobs": 1,
                "param_grid": {
                    "nohe__sparse": [
                        true,
                        false
                    ],
                    "rfc__N_estimators": [
                        10,
                        100
                    ],
                    "rfc__max_depth": [
                        4,
                        6,
                        8
                    ],
                    "rfc__max_leaf_nodes": [
                        3,
                        6,
                        9
                    ]
                },
                "pre_dispatch": "2*n_jobs",
                "refit": true,
                "scoring": null,
                "verbose": 10
            },
            "E. Estimator": {
                "A. label": "test_complex nested_estimator",
                "B. transformer": "<class 'sklearn.pipeline.Pipeline'>",
                "C. children": [
                    {
                        "A. label": "nfu",
                        "B. transformer": "<class 'sklearn.pipeline.FeatureUnion'>",
                        "C. children": [
                            {
                                "A. label": "nce",
                                "B. transformer": "<class 'data_preparation.transformers.ColumnExtractor'>",
                                "D. vars": {
                                    "cols": [
                                        "v22"
                                    ]
                                }
                            },
                            {
                                "A. label": "nohe",
                                "B. transformer": "<class 'sklearn.preprocessing.data.OneHotEncoder'>",
                                "D. vars": {
                                    "categorical_features": "all",
                                    "dtype": "<type 'float'>",
                                    "handle_unknown": "error",
                                    "n_values": "auto",
                                    "sparse": true
                                }
                            }
                        ]
                    },
                    {
                        "A. label": "rfc",
                        "B. transformer": "<class 'sklearn.ensemble.forest.RandomForestClassifier'>",
                        "D. vars": {
                            "base_estimator": "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,\n            min_samples_split=2, min_weight_fraction_leaf=0.0,\n            presort=False, random_state=None, splitter='best')",
                            "bootstrap": true,
                            "class_weight": null,
                            "criterion": "gini",
                            "estimator_params": [
                                "criterion",
                                "max_depth",
                                "min_samples_split",
                                "min_samples_leaf",
                                "min_weight_fraction_leaf",
                                "max_features",
                                "max_leaf_nodes",
                                "random_state"
                            ],
                            "estimators_": [],
                            "max_depth": null,
                            "max_features": "auto",
                            "max_leaf_nodes": null,
                            "min_samples_leaf": 1,
                            "min_samples_split": 2,
                            "min_weight_fraction_leaf": 0.0,
                            "n_estimators": 10,
                            "n_jobs": 1,
                            "oob_score": false,
                            "random_state": null,
                            "verbose": 0,
                            "warm_start": false
                        }
                    }
                ]
            }
        }

        """
        self.assertEqual(expected_result, result, "complex transformer doesn't match")
