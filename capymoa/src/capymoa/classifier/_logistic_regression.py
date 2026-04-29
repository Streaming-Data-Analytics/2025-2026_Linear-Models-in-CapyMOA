from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

import moa.classifiers as moa_classifiers

class LogisticRegression(MOAClassifier):
    """Logistic Regression.

    Logistic Regression is a linear classifier for binary classification, trained incrementally
    using Stochastic Gradient Descent (SGD). It optimizes the log loss and allows for both L1 and
    L2 regularization. The gradient can be clipped to avoid the exploding gradient problem.

    >>> from capymoa.classifier import LogisticRegression
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = LogisticRegression(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    """

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
        """Construct Logistic Regression.

        :param schema: Stream schema.
        :param random_seed: Seed for reproducibility (not used by this classifier).
        :param learning_rate: The learning rate for the SGD optimizer.
        :param bias_learning_rate: The learning rate for the intercept (bias).
        :param l1_penalty: The L1 regularization weight.
        :param l2_penalty: The L2 regularization weight.
        :param clip_gradient: Maximum magnitude for gradients (gradient clipping).
        :param bias_init: Initial value for the intercept (bias).
        """
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
            moa_learner=moa_classifiers.LogisticRegression,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )

    def __str__(self):
        return "LogisticRegression"
