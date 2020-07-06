import os


def training_partition_path(output_dir: str, tomo_name: str, partition_name: str):
    output_path_dir = os.path.join(output_dir, "training_data")
    output_path_dir = os.path.join(output_path_dir, tomo_name)
    output_h5_file_name = partition_name + ".h5"
    output_path = os.path.join(output_path_dir, output_h5_file_name)
    return output_path_dir, output_path


def testing_partition_path(output_dir: str, tomo_name: str, model_name: str, partition_name: str):
    output_path_dir = os.path.join(output_dir, "test_partitions")
    output_path_dir = os.path.join(output_path_dir, tomo_name)
    output_path_dir = os.path.join(output_path_dir, model_name)
    output_h5_file_name = partition_name + ".h5"
    output_path = os.path.join(output_path_dir, output_h5_file_name)
    return output_path_dir, output_path