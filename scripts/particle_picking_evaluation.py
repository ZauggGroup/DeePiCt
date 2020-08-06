import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-class_number", "--class_number", type=int)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-calculate_motl", "--calculate_motl", type=str)
parser.add_argument("-statistics_file", "--statistics_file", type=str)
parser.add_argument("-radius", "--radius", type=int)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.csv import motl_writer
from file_actions.writers.csv import write_statistics
from networks.utils import build_prediction_output_dir
from performance.statistics_utils import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values

tomo_name = args.tomo_name
output_dir = args.output_dir
model_name = os.path.basename(args.model_name)[:-4]
class_number = args.class_number
dataset_table = args.dataset_table
statistics_file = args.statistics_file
radius = args.radius

segmentation_label = os.path.basename(model_name)

semantic_classes = args.segmentation_names
semantic_class = semantic_classes[class_number]

DTHeader = DatasetTableHeader(semantic_classes=semantic_classes)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

print("Processing tomo", tomo_name)
output_dir = os.path.join(output_dir, "predictions")
tomo_output_dir = build_prediction_output_dir(base_output_dir=output_dir,
                                              label_name="", model_name=segmentation_label,
                                              tomo_name=tomo_name, semantic_class=semantic_class)
print(tomo_output_dir)
os.makedirs(tomo_output_dir, exist_ok=True)
motl_in_dir = [file for file in os.listdir(tomo_output_dir) if 'motl_' in file]
assert len(motl_in_dir) == 1, "only one motive list can be filtered."
path_to_motl_predicted = os.path.join(tomo_output_dir, motl_in_dir[0])

tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
path_to_motl_true = tomo_df.iloc[0][DTHeader.clean_motls[class_number]]

class_name = semantic_classes[class_number]

figures_dir = os.path.join(tomo_output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
    path_to_motl=path_to_motl_predicted)

true_values, true_coordinates = read_motl_coordinates_and_values(
    path_to_motl=path_to_motl_true)
unique_peaks_number = len(predicted_values)

predicted_coordinates = np.array(predicted_coordinates)

prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, *_ = \
    precision_recall_calculator(
        predicted_coordinates=predicted_coordinates,
        value_predicted=predicted_values,
        true_coordinates=true_coordinates,
        radius=radius)

F1_score = f1_score_calculator(prec, recall)

if len(F1_score) > 0:
    max_F1 = np.max(F1_score)
    optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
else:
    max_F1 = 0
    optimal_peak_number = np.nan

auPRC = pr_auc_score(precision=prec, recall=recall)
print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

tomo_output_dir = os.path.join(tomo_output_dir, "pr_radius_" + str(radius))
path_to_detected_predicted = join(tomo_output_dir, "detected")
path_to_undetected_predicted = join(tomo_output_dir, "undetected")

os.makedirs(path_to_detected_predicted, exist_ok=True)
os.makedirs(path_to_undetected_predicted, exist_ok=True)

motl_writer(path_to_output_folder=path_to_detected_predicted,
            list_of_peak_coords=tp_pred,
            list_of_peak_scores=tp_pred_scores,
            in_tom_format=True)
motl_writer(path_to_output_folder=path_to_undetected_predicted,
            list_of_peak_coords=fp_pred,
            list_of_peak_scores=fp_pred_scores,
            in_tom_format=True)
matplotlib.use('Agg')
plt.figure(1)
plt.hist(predicted_values, bins=45, label="predicted")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(predicted_values)) + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram_predicted.png")
plt.savefig(fname=fig_name, format="png")

plt.figure(2)
plt.hist(tp_pred_scores, bins=45, label="true positives", fc=(0, 0, 1, 0.5))
plt.hist(fp_pred_scores, bins=45, label="false positives", fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(fp_pred)) + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram-detected-undetected.png")
plt.savefig(fname=fig_name, format="png")

plt.figure(3)
pr_legend_str = "auPRC = " + str(round(auPRC, 4))
f1_legend_str = "(max_F1, best_peaks) = ({}, {})".format(round(max_F1, 4), optimal_peak_number)
title_str = str(len(predicted_coordinates)) + " peaks"
plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title(title_str)
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
plt.savefig(fname=fig_name, format="png")

plt.figure(4)
plt.plot(recall, prec, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(title_str)
plt.legend()
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.gcf()
fig_name = join(figures_dir, "pr_" + title_str + ".png")
plt.savefig(fname=fig_name, format="png")
plt.close()

print("statistics_file", statistics_file)
if statistics_file != "None":
    statistics_label = segmentation_label + "_pr_radius_" + str(radius)

    write_statistics(statistics_file=statistics_file,
                     statistics_label="auPRC_" + statistics_label,
                     tomo_name=tomo_name,
                     stat_measure=round(auPRC, 4))

    write_statistics(statistics_file=statistics_file,
                     statistics_label="F1_" + statistics_label,
                     tomo_name=tomo_name,
                     stat_measure=round(max_F1, 4))

### For snakemake:
snakemake_pattern = os.path.join(path_to_detected_predicted, ".done_pp_snakemake")
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
