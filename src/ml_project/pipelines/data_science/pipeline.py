"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.4
"""

from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["train_x", "train_data_y", "parameters"],
                "example_model",
            ),
            node(
                predict,
                dict(model="example_model", test_x="test_x"),
                "example_predictions",
            ),
            node(report_accuracy, ["example_predictions", "test_data_y"], None),
        ]
    )
