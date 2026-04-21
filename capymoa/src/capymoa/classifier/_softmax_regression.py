from __future__ import annotations
import typing

from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

import moa.classifiers as moa_classifiers


class SoftmaxRegression(MOAClassifier):
    """Softmax Regression.

    Softmax Regression is an online classifier for multi-class classification.
    It performs multinomial logistic regression by minimizing the 
    cross-entropy loss using Stochastic Gradient Descent (SGD). It applies
    a softmax activation function to produce class probabilities.

    >>> from capymoa.classifier import SoftmaxRegression
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = SoftmaxRegression(stream.get_schema(), learning_rate=0.01)
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        random_seed: int = 0,
        learning_rate: float = 0.01,
        bias_learning_rate: float = 0.01,
        l2_penalty: float = 0.0,
        bias_init: float = 0.0,
    ):
        """Construct Softmax Regression.

        :param schema: Stream schema.
        :param random_seed: Seed for reproducibility.
        :param learning_rate: The learning rate for the SGD optimizer.
        :param bias_learning_rate: The learning rate for the intercept (bias).
        :param l2_penalty: The L2 regularization weight.
        :param bias_init: Initial value for the intercept (bias).
        """
        mapping = {
            "learning_rate": "-r",
            "bias_learning_rate": "-b",
            "l2_penalty": "-l",
            "bias_init": "-i",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        # Initialize the parent MOAClassifier
        super(SoftmaxRegression, self).__init__(
            moa_learner=moa_classifiers.SoftmaxRegression,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )

    def __str__(self):
        return "SoftmaxRegression CapyMOA Classifier"
