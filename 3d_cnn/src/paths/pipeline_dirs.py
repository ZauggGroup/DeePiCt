import os


def partition_path(output_dir: str, tomo_name: str, partition_name: str, fold: int or None = None):
    output_path_dir = os.path.join(output_dir, partition_name)
    output_path_dir = os.path.join(output_path_dir, tomo_name)
    if fold is not None:
        output_path_dir = os.path.join(output_path_dir, "fold_" + str(fold))
    output_h5_file_name = "partition.h5"
    output_path = os.path.join(output_path_dir, output_h5_file_name)
    return output_path_dir, output_path


def training_partition_path(output_dir: str, tomo_name: str, fold: int or None = None):
    output_path_dir, output_path = partition_path(output_dir=output_dir, tomo_name=tomo_name,
                                                  partition_name="training_data", fold=fold)
    return output_path_dir, output_path


def testing_partition_path(output_dir: str, tomo_name: str, fold: int or None = None):
    output_path_dir, output_path = partition_path(output_dir=output_dir, tomo_name=tomo_name,
                                                  partition_name="testing_data", fold=fold)
    return output_path_dir, output_path


def fold_testing_partition_path(output_dir: str, tomo_name: str, model_name: str, partition_name: str, fold: str):
    model_name = os.path.basename(model_name)
    output_path_dir = os.path.join(output_dir, "test_partitions")
    output_path_dir = os.path.join(output_path_dir, tomo_name)
    output_path_dir = os.path.join(output_path_dir, model_name)
    output_h5_file_name = partition_name + fold + ".h5"
    output_path = os.path.join(output_path_dir, output_h5_file_name)
    return output_path_dir, output_path


def get_probability_map_path(output_dir: str, model_name: str, tomo_name: str, semantic_class: str):
    model_name = os.path.basename(model_name)
    tomo_output_dir = os.path.join(output_dir, "predictions")
    tomo_output_dir = os.path.join(tomo_output_dir, model_name)
    tomo_output_dir = os.path.join(tomo_output_dir, tomo_name)
    tomo_output_dir = os.path.join(tomo_output_dir, semantic_class)
    output_path = os.path.join(tomo_output_dir, "probability_map.mrc")
    return tomo_output_dir, output_path


def get_post_processed_prediction_path(output_dir: str, model_name: str, tomo_name: str, semantic_class: str):
    model_name = os.path.basename(model_name)
    tomo_output_dir, _ = get_probability_map_path(output_dir=output_dir, model_name=model_name, tomo_name=tomo_name,
                                                  semantic_class=semantic_class)
    post_processed_output_path = os.path.join(tomo_output_dir, "post_processed_prediction.mrc")
    return post_processed_output_path


def get_models_table_path(output_dir) -> str:
    models_table = os.path.join(output_dir, "models")
    models_table = os.path.join(models_table, "models.csv")
    return models_table
