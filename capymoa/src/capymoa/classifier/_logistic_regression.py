from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

import moa.classifiers as moa_classifiers

#TODO: add comments
class LogisticRegression(MOAClassifier):

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0, # not used
        learning_rate: float = 0.01,
        bias_learning_rate: float = 0.01,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        clip_gradient: float = 1e12,
        bias_init: float = 0.0
    ):
        mapping = {
            "learning_rate": "-r",
            "bias_learning_rate": "-b",
            "l1_penalty": "-l",
            "l2_penalty": "-q",
            "clip_gradient": "-c",
            "bias_init": "-i"
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super(LogisticRegression, self).__init__(
            moa_learner=moa_classifiers.LogisticRegressionClassifier,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )