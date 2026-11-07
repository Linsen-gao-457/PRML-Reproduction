import numpy as np
import pytest

from prml.classification import PerceptronClassifier


@pytest.fixture
def figure_4_7_data():
    features = np.array(
        [
            [0.24, 0.72],
            [0.46, 0.30],
            [0.94, 0.90],
            [0.37, -0.48],
            [0.66, -0.89],
            [-0.75, 0.95],
            [-0.78, 0.60],
            [-0.55, 0.64],
            [-0.42, 0.72],
            [-0.10, -0.20],
        ]
    )
    targets = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    return features, targets


def test_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="learning_rate"):
        PerceptronClassifier(learning_rate=0.0)

    with pytest.raises(ValueError, match="max_epochs"):
        PerceptronClassifier(max_epochs=0)


def test_figure_4_7_converges_after_two_updates(figure_4_7_data):
    features, targets = figure_4_7_data
    model = PerceptronClassifier(max_epochs=100)

    returned_model = model.fit(
        featrues=features,
        targets=targets,
        initial_weight=[-0.40, -0.90],
    )

    assert returned_model is model
    assert model.converged_
    assert model.n_epochs_ == 2
    assert len(model.history_) == 2
    np.testing.assert_allclose(model.coef_, [0.30, 0.12])
    assert model.intercept_ == 0.0


def test_history_records_expected_weight_updates(figure_4_7_data):
    features, targets = figure_4_7_data
    model = PerceptronClassifier().fit(
        featrues=features,
        targets=targets,
        initial_weight=[-0.40, -0.90],
    )

    first_update, second_update = model.history_

    assert first_update["epoch"] == 0
    assert first_update["index"] == 0
    np.testing.assert_allclose(first_update["feature"], [0.24, 0.72])
    np.testing.assert_allclose(first_update["weights_before"], [-0.40, -0.90])
    np.testing.assert_allclose(first_update["update"], [0.24, 0.72])
    np.testing.assert_allclose(first_update["weights_after"], [-0.16, -0.18])

    assert second_update["epoch"] == 0
    assert second_update["index"] == 1
    np.testing.assert_allclose(second_update["feature"], [0.46, 0.30])
    np.testing.assert_allclose(second_update["weights_before"], [-0.16, -0.18])
    np.testing.assert_allclose(second_update["update"], [0.46, 0.30])
    np.testing.assert_allclose(second_update["weights_after"], [0.30, 0.12])


def test_fitted_scores_separate_training_classes(figure_4_7_data):
    features, targets = figure_4_7_data
    model = PerceptronClassifier().fit(
        featrues=features,
        targets=targets,
        initial_weight=[-0.40, -0.90],
    )

    scores = model.decision_function(features)

    assert scores.shape == targets.shape
    assert np.all(scores[targets == 1] > 0.0)
    assert np.all(scores[targets == -1] < 0.0)


def test_supports_non_numeric_class_labels():
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    targets = np.array(["blue", "blue", "red", "red"])

    model = PerceptronClassifier().fit(
        featrues=features,
        targets=targets,
    )

    scores = model.decision_function(features)
    signed_targets = np.where(targets == model.classes_[0], -1.0, 1.0)
    assert np.all(signed_targets * scores > 0.0)


def test_decision_function_requires_fitted_model():
    model = PerceptronClassifier()

    with pytest.raises(RuntimeError, match="fit"):
        model.decision_function([[1.0, 2.0]])


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        ([1.0, 2.0], [0], "two-dimensional"),
        ([[1.0], [2.0]], [[0], [1]], "one-dimensional"),
        ([[1.0], [2.0]], [0], "same number"),
        ([], [], "two-dimensional"),
        ([[1.0], [np.nan]], [0, 1], "finite"),
    ],
)
def test_rejects_invalid_training_data(features, targets, message):
    model = PerceptronClassifier()

    with pytest.raises(ValueError, match=message):
        model.fit(
            featrues=features,
            targets=targets,
        )


def test_requires_exactly_two_classes():
    model = PerceptronClassifier()

    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit(
            featrues=[[1.0], [2.0], [3.0]],
            targets=[0, 1, 2],
        )


def test_rejects_invalid_initial_weight(figure_4_7_data):
    features, targets = figure_4_7_data
    model = PerceptronClassifier()

    with pytest.raises(ValueError, match="wrong shape"):
        model.fit(
            featrues=features,
            targets=targets,
            initial_weight=[0.0],
        )

    with pytest.raises(ValueError, match="finite"):
        model.fit(
            featrues=features,
            targets=targets,
            initial_weight=[0.0, np.inf],
        )
