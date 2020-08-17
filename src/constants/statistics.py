import os

import pandas as pd

from constants.dataset_tables import ModelsTableHeader
from dataclasses import dataclass


@dataclass
class ModelPerformanceVector:
    """Class for keeping track of model performance"""

    batch_norm: bool
    box_size: int
    training_date: str
    decoder_dropout: float
    encoder_dropout: float
    depth: int
    initial_features: int
    label_name: str
    log_path: str
    model_name: str
    model_path: str
    n_epochs: int
    old_model: None or str
    output_classes: int
    overlap: int
    partition_name: str
    processing_tomo: str
    retrain: bool
    segmentation_names: str
    train_split: float
    training_set: str
    tomo_name: str
    fold: int or None
    pr_radius: float
    min_cluster_size: int
    max_cluster_size: int
    threshold: float
    statistic_variable: str
    statistic_value: float
    prediction_class: str


def get_model_performance_data(models_table, model_name, tomo_name, statistic_variable, statistic_value, pr_radius,
                               cv_fold, min_cluster_size, max_cluster_size, threshold, prediction_class):
    ModelsHeader = ModelsTableHeader()
    models_df = pd.read_csv(models_table, dtype={ModelsHeader.model_name: str,
                                                 ModelsHeader.semantic_classes: str})

    model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
    assert model_df.shape[0] < 2, "several models have the same name in models.csv"
    assert model_df.shape[0] > 0, "no model with name " + model_name + " in models.csv"

    batch_norm = model_df.iloc[0][ModelsHeader.batch_normalization]
    model_path = model_df.iloc[0][ModelsHeader.model_path]
    overlap = model_df.iloc[0][ModelsHeader.overlap]
    box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
    semantic_names = model_df.iloc[0][ModelsHeader.semantic_classes]
    training_date = model_df.iloc[0][ModelsHeader.date]
    decoder_dropout = model_df.iloc[0][ModelsHeader.decoder_dropout]
    encoder_dropout = model_df.iloc[0][ModelsHeader.encoder_dropout]
    depth = model_df.iloc[0][ModelsHeader.depth]
    initial_features = model_df.iloc[0][ModelsHeader.initial_features]
    label_name = model_df.iloc[0][ModelsHeader.label_name]
    log_path = model_df.iloc[0][ModelsHeader.logging_path]
    epochs = model_df.iloc[0][ModelsHeader.epochs]
    old_model = model_df.iloc[0][ModelsHeader.old_model]
    output_classes = model_df.iloc[0][ModelsHeader.output_classes]
    partition_data = model_df.iloc[0][ModelsHeader.partition_name]
    processing_tomo = model_df.iloc[0][ModelsHeader.processing_tomo]
    retrain = model_df.iloc[0][ModelsHeader.retrain]
    train_split = model_df.iloc[0][ModelsHeader.train_split]
    training_set = model_df.iloc[0][ModelsHeader.training_set]

    model_performance_vector = ModelPerformanceVector(batch_norm=batch_norm,
                                                      box_size=box_shape,
                                                      training_date=training_date,
                                                      decoder_dropout=decoder_dropout,
                                                      encoder_dropout=encoder_dropout,
                                                      depth=depth,
                                                      initial_features=initial_features,
                                                      label_name=label_name,
                                                      log_path=log_path,
                                                      model_name=model_name,
                                                      model_path=model_path,
                                                      n_epochs=epochs,
                                                      old_model=old_model,
                                                      output_classes=output_classes,
                                                      overlap=overlap,
                                                      partition_name=partition_data,
                                                      processing_tomo=processing_tomo,
                                                      retrain=retrain,
                                                      segmentation_names=semantic_names,
                                                      train_split=train_split,
                                                      training_set=training_set,
                                                      tomo_name=tomo_name,
                                                      fold=cv_fold,
                                                      pr_radius=pr_radius,
                                                      min_cluster_size=min_cluster_size,
                                                      max_cluster_size=max_cluster_size,
                                                      threshold=threshold,
                                                      statistic_variable=statistic_variable,
                                                      statistic_value=statistic_value,
                                                      prediction_class=prediction_class)
    return model_performance_vector


def add_model_performance_statistics(model_performance_vector, file):
    data = model_performance_vector.__dict__
    model_performance_df = pd.DataFrame.from_dict([data])
    if os.path.exists(file):
        statistics_df = pd.read_csv(file)
        print("Statistics table exists, with shape:", statistics_df.shape)
        if statistics_df.shape[0] > 0:
            statistics_df = statistics_df.append(model_performance_df, sort="False")
            statistics_df.to_csv(file, index=False)
        else:
            model_performance_df.to_csv(file, index=False)
    else:
        model_performance_df.to_csv(file, index=False)
    return


def write_statistics_pp(statistics_file, tomo_name, model_name, models_table_path, statistic_variable,
                        statistic_value, pr_radius, cv_fold, min_cluster_size, max_cluster_size, threshold,
                        prediction_class):
    model_performance_vector = get_model_performance_data(models_table=models_table_path, model_name=model_name,
                                                          tomo_name=tomo_name, statistic_variable=statistic_variable,
                                                          statistic_value=statistic_value, pr_radius=pr_radius,
                                                          cv_fold=cv_fold, min_cluster_size=min_cluster_size,
                                                          max_cluster_size=max_cluster_size, threshold=threshold,
                                                          prediction_class=prediction_class)
    add_model_performance_statistics(model_performance_vector=model_performance_vector, file=statistics_file)
    return
