import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class ModelDescriptor:
    """Class for keeping track of model architecture"""
    batch_norm: bool
    box_size: int
    training_date: str
    decoder_dropout: float
    encoder_dropout: float
    depth: int
    initial_features: int
    log_path: str
    model_name: str
    model_path: str
    epochs: int
    old_model: None or str
    output_classes: int
    overlap: int
    partition_name: str
    processing_tomo: str
    retrain: bool
    semantic_classes: str
    train_split: float
    training_set: list
    testing_set: list or None
    total_folds: int or None
    fold: int or None
    da_rounds: int
    da_rot_angle: float
    da_elastic_alpha: float
    da_sigma_gauss: float
    da_salt_pepper_p: float
    da_salt_pepper_ampl: float

    @staticmethod
    def from_data_frame(df: pd.DataFrame) -> "ModelDescriptor":
        assert df.shape[0] == 1
        dict_df = {col: df.iloc[0][col] for col in df.columns}
        model_descriptor = ModelDescriptor(**dict_df)
        return model_descriptor

    @staticmethod
    def to_data_frame(model_descriptor: "ModelDescriptor"):
        return pd.DataFrame([model_descriptor.__dict__])

    @staticmethod
    def add_descriptor_to_table(table_path, model_descriptor, force_retrain=False) -> None:
        model_descriptor_row = model_descriptor.to_data_frame(model_descriptor)
        models_notebook_dir = os.path.dirname(table_path)
        os.makedirs(models_notebook_dir, exist_ok=True)
        if os.path.isfile(table_path):
            models_table = pd.read_csv(table_path)
            print("model names in table:", models_table["model_name"].values)
            print("model_descriptor.model_name:", model_descriptor.model_name)
            if model_descriptor.model_name in models_table["model_name"].values:
                select_indices = list(np.where(models_table["model_name"] == model_descriptor.model_name)[0])
                models_table = models_table.drop(select_indices)
                models_table = models_table.append(model_descriptor_row, sort="False")
            else:
                models_table = models_table.append(model_descriptor_row, sort="False")
            models_table.to_csv(path_or_buf=table_path, index=False)
        else:
            model_descriptor_row.to_csv(path_or_buf=table_path, index=False)
        return


@dataclass
class PerformanceVector:
    """Class to keep track of performance results"""
    tomo_name: str
    pr_radius: float
    min_cluster_size: float
    max_cluster_size: float
    clustering_connectivity: int
    threshold: float
    statistic_variable: str
    statistic_value: float
    prediction_class: str


class ModelPerformanceVector:
    """Class for keeping track of model performance"""

    def __init__(self, model_descriptor: ModelDescriptor, performance_dict: dict):
        self.model_descriptor = model_descriptor
        self.model_performance = PerformanceVector(**performance_dict)


def add_model_performance_statistics(model_performance_vector, file):
    data = model_performance_vector.model_descriptor.__dict__
    data.update(model_performance_vector.model_performance.__dict__)
    # noinspection PyTypeChecker
    model_performance_row = pd.DataFrame.from_dict([data])
    if os.path.exists(file):
        statistics_df = pd.read_csv(file)
        print("Statistics table exists, with shape:", statistics_df.shape)
        if statistics_df.shape[0] > 0:
            statistics_df = statistics_df.append(model_performance_row, sort="False")
            statistics_df.to_csv(file, index=False)
        else:
            model_performance_row.to_csv(file, index=False)
    else:
        model_performance_row.to_csv(file, index=False)
    return


def write_statistics_pp(statistics_file, tomo_name, model_descriptor: ModelDescriptor, statistic_variable,
                        statistic_value, pr_radius, min_cluster_size, max_cluster_size, threshold,
                        prediction_class, clustering_connectivity):
    performance_dict = {"tomo_name": tomo_name, "pr_radius": pr_radius,
                        "min_cluster_size": min_cluster_size,
                        "max_cluster_size": max_cluster_size,
                        "clustering_connectivity": clustering_connectivity,
                        "threshold": threshold,
                        "statistic_variable": statistic_variable,
                        "statistic_value": statistic_value,
                        "prediction_class": prediction_class}

    model_performance_vector = ModelPerformanceVector(performance_dict=performance_dict,
                                                      model_descriptor=model_descriptor)
    add_model_performance_statistics(model_performance_vector=model_performance_vector, file=statistics_file)
    return
