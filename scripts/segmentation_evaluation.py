import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-statistics_file", "--statistics_file", type=str)
parser.add_argument("-class_number", "--class_number", type=int)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-filtering_mask", "--filtering_mask", type=str)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)
import os

import numpy as np
import pandas as pd
import torch, yaml

from constants.dataset_tables import DatasetTableHeader, ModelsTableHeader
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import write_statistics
from pytorch_cnn.classes.loss import DiceCoefficient
from paths.pipeline_dirs import get_post_processed_prediction_path

args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))

dataset_table = args.dataset_table
tomo_name = args.tomo_name
output_dir = args.output_dir
class_number = args.class_number
statistics_file = args.statistics_file
filtering_mask = args.filtering_mask
model_name = os.path.basename(args.model_name)[:-4]
label_name = model_name

semantic_classes = args.segmentation_names
semantic_class = semantic_classes[class_number]

prediction_path = get_post_processed_prediction_path(output_dir=output_dir,
                                                     model_name=model_name,
                                                     tomo_name=tomo_name,
                                                     semantic_class=semantic_class)
print(prediction_path)
assert os.path.isfile(prediction_path), "The prediction file does not exist!"

DTHeader = DatasetTableHeader(semantic_classes=semantic_classes, filtering_mask=filtering_mask)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
clean_mask_name = DTHeader.masks_names[class_number]

tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
lamella_file = tomo_df.iloc[0][DTHeader.filtering_mask]
target_path = tomo_df.iloc[0][clean_mask_name]

if str(lamella_file) == "nan":
    prediction = load_tomogram(prediction_path)
else:
    lamella_indicator = load_tomogram(path_to_dataset=lamella_file)
    prediction = load_tomogram(path_to_dataset=prediction_path)
    shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                     zip(lamella_indicator.shape, prediction.shape)]
    lamella_indicator = lamella_indicator[:shx, :shy, :shz]
    prediction = prediction[:shx, :shy, :shz]
    prediction = np.array(lamella_indicator, dtype=float) * np.array(prediction,
                                                                     dtype=float)

target = load_tomogram(path_to_dataset=target_path)

shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                 zip(target.shape, prediction.shape)]

target = target[:shx, :shy, :shz]
prediction = prediction[:shx, :shy, :shz]

prediction = torch.from_numpy(prediction).float()
target = torch.from_numpy(target).float()

measure = DiceCoefficient()
dice_statistic = measure.forward(prediction=prediction, target=target)
dice_statistic = float(dice_statistic)
statistics_label = label_name + "_dice"

print("statistics_file", statistics_file)
if statistics_file != "None":
    write_statistics(statistics_file=statistics_file,
                     statistics_label=statistics_label,
                     tomo_name=tomo_name,
                     stat_measure=dice_statistic)

from constants.statistics import write_statistics_pp
models_table = os.path.join(args.output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
fold = None
threshold = config["postprocessing_clustering"]["threshold"]
min_cluster_size = config["postprocessing_clustering"]["min_cluster_size"]
max_cluster_size = config["postprocessing_clustering"]["max_cluster_size"]
if max_cluster_size is None:
    max_cluster_size = np.inf

if os.path.dirname(statistics_file) == "":
    statistics_file = "pp_statistics.csv"
    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                        models_table_path=models_table, statistic_variable="dice",
                        statistic_value=round(dice_statistic, 4), pr_radius=None, cv_fold=fold,
                        min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                        threshold=threshold, prediction_class=semantic_class)
else:
    statistics_file = os.path.dirname(statistics_file) + "/pp_statistics.csv"
    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                        models_table_path=models_table, statistic_variable="dice",
                        statistic_value=round(dice_statistic, 4), pr_radius=None, cv_fold=fold,
                        min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                        threshold=threshold, prediction_class=semantic_class)

print("Dice coefficient =", dice_statistic)

### For snakemake:
prediction_dir = os.path.dirname(prediction_path)
snakemake_pattern = os.path.join(prediction_dir, ".done_dice_eval_snakemake")
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
