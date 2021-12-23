import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")
parser.add_argument("-config_file", "--config_file", help="yaml_file", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
import ast

import pandas as pd

from constants.config import Config
from networks.utils import build_prediction_output_dir
from constants.config import get_model_name

config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)

model_path, model_name = get_model_name(config, fold)
# snakemake_pattern = config.output_dir + "/predictions/." + model_name + "/" + \
#                     config.pred_class + "/.{fold}.global_eval_snakemake".format(fold=str(fold))
from networks.utils import get_training_testing_lists

if isinstance(fold, int):
    tomo_training_list, tomo_testing_list = get_training_testing_lists(config=config, fold=fold)
    if len(tomo_testing_list) > 0:
        print("model_name", model_name)
        stats_frames = []
        for tomo_name in tomo_testing_list:
            pred_output_dir = os.path.join(config.output_dir, "predictions")
            tomo_output_dir = build_prediction_output_dir(base_output_dir=pred_output_dir,
                                                          label_name="", model_name=model_name,
                                                          tomo_name=tomo_name, semantic_class=config.pred_class)
            print(tomo_output_dir)

            in_statistics_file = pd.read_csv(tomo_output_dir, "pp_statistics.csv")
            stats_frames.append(in_statistics_file)

        out_statistics_file = os.path.join(config.output_dir, "pp_statistics.csv")
        if os.path.isfile(out_statistics_file):
            prev_stats = pd.read_csv(out_statistics_file)
            stats_frames.append(prev_stats)
        tmp_statistics = pd.concat(stats_frames, axis=0)

# For snakemake:
# snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
# os.makedirs(snakemake_pattern_dir, exist_ok=True)
# with open(file=snakemake_pattern, mode="w") as f:
#     print("Creating snakemake pattern", snakemake_pattern)
