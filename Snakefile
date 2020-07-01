"""
Modular pipeline for Spatial Point Pattern Analysis in 3D
"""

__author__ = "Irene de Teresa"
__version__ = "0.1.0"
__maintainer__ = "Irene de Teresa"
__email__ = "irene.de.teresa@embl.de"

import os, yaml
import os.path as op
import pandas as pd
from json import dumps
from datetime import datetime
from warnings import warn

# Load user's config file
cli_config = config.copy()
user_config_file = cli_config["config"]
with open(user_config_file, 'r') as user_config:
    config = yaml.safe_load(user_config)

srcdir = os.path.join(workflow.basedir, "src")
scriptdir = os.path.join(workflow.basedir, "scripts")

run_name = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")


# General data
dataset_table = config["dataset_table"]
output_dir = config["output_dir"]
model_name = config["model_name"]

# Tomogram lists
training_tomos = config["tomos_sets"]["training_list"]
prediction_tomos = config["tomos_sets"]["prediction_list"]

# Training
overlap = config["training"]["overlap"]
partition_name = config["training"]["partition_name"]
segmentation_names = config["training"]["semantic_classes"]
processing_tomo = config["training"]["processing_tomo"]
box_shape = config["training"]["box_shape"]
min_label_fraction = config["training"]["min_label_fraction"]
max_label_fraction = config["training"]["max_label_fraction"]

#unet_hyperparameters:
depth = config["training"]["unet_hyperparameters"]["depth"]
initial_features = config["training"]["unet_hyperparameters"]["initial_features"]
epochs = config["training"]["unet_hyperparameters"]["epochs"]
split = config["training"]["unet_hyperparameters"]["split"]
BatchNorm = config["training"]["unet_hyperparameters"]["BatchNorm"]
encoder_dropout = config["training"]["unet_hyperparameters"]["encoder_dropout"]
decoder_dropout = config["training"]["unet_hyperparameters"]["decoder_dropout"]
batch_size = config["training"]["unet_hyperparameters"]["batch_size"]

# prediction:
pred_active = config["prediction"]["active"]
pred_processing_tomo = config["prediction"]["processing_tomo"]
pred_partition_name = config["prediction"]["partition_name"]
pred_class_number = config["prediction"]["class_number"]

# Thresholding clustering and motl generation
# postprocessing:
#   clustering_parameters:
postproc_active = config["postprocessing"]["clustering_parameters"]
threshold = config["postprocessing"]["clustering_parameters"]
min_cluster_size = config["postprocessing"]["clustering_parameters"]
max_cluster_size = config["postprocessing"]["clustering_parameters"]
calculate_motl = config["postprocessing"]["clustering_parameters"]
ignore_border_thickness = config["postprocessing"]["clustering_parameters"]
filtering_mask = config["postprocessing"]["clustering_parameters"]

# # For precision recall in particle picking
# evaluation:
active = config["evaluation"]["active"]
pr_tolerance_radius = config["evaluation"]["pr_tolerance_radius"]
statistics_file = config["evaluation"]["statistics_file"]

tomo_name_training = expand(["{tomo_name}"], tomo_name=training_tomos)
done_training_set_generation_pattern = expand([".snakemake/{run_name}_training_set_generation_{tomo}.done"],
                                              tomo=training_tomos, run_name=run_name)
done_training_pattern = expand([".snakemake/{run_name}_3d_training_{tomo}.done"], tomo=training_tomos,
                               run_name=run_name)
# done_prediction_set_generation_pattern = expand([".snakemake/{run_name}_prediction_set_generation_{tomo}.done"],
#                                                 tomo=training_tomos, run_name=run_name)
# done_segmenting_pattern = expand([".snakemake/{run_name}_segmenting_{tomo}.done"], tomo=training_tomos,
#                                  run_name=run_name)
# done_assembling_segmentation_pattern = expand([".snakemake/{run_name}_assembling_segmentation_{tomo}.done"],
#                                               tomo=training_tomos, run_name=run_name)

str_segmentation_names = ""
for name in segmentation_names:
    str_segmentation_names += name + " "

targets = []
# Rules
# rule all:
#     shell:
#         "echo 'Starting pipeline.'"

rule training_set_generation:
    conda:
        "environment.yaml"
    # log: "test.log"
    output:
          done_training_set_generation_pattern
    shell:
         f"""
        python3 {scriptdir}/generate_training_data.py \
        --pythonpath {srcdir} \
        --overlap {{overlap}}  \
        --partition_name {{partition_name}} \
        --segmentation_names {{str_segmentation_names}} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --processing_tomo {{processing_tomo}} \
        --box_shape {{box_shape}} \
        --min_label_fraction {{min_label_fraction}} \
        --max_label_fraction {{max_label_fraction}} \
        --tomo_name {{tomo_name_training}} \
        && touch {{done_training_set_generation_pattern}}
        """

# rule training_3dunet:
#     conda:
#         "environment.yaml"
#     # log: "test.log"
#     output:
#           done_training_pattern
#     shell:
#          f"""
#         python3 {scriptdir}/generate_training_data.py \
#         --pythonpath {srcdir} \
#         --overlap {{overlap}}  \
#         --partition_name {{partition_name}} \
#         --segmentation_names {{str_segmentation_names}} \
#         --dataset_table {{dataset_table}} \
#         --output_dir {{output_dir}} \
#         --processing_tomo {{processing_tomo}} \
#         --box_shape {{box_shape}} \
#         --min_label_fraction {{min_label_fraction}} \
#         --max_label_fraction {{max_label_fraction}} \
#         --tomo_name {{tomo_name_training}} \
#         && touch {{done_training_pattern}}
#         """
# -pythonpath
# -partition_name
# -segmentation_names
# -dataset_table
# -output_dir
# -box_shape
# -gpu
# logging_dir
# model_dir
# models_table
# tomo_training_list
# split
# n_epochs
# depth
# decoder_dropout
# encoder_dropout
# batch_size
# batch_norm
# initial_features
# overlap
# processing_tomo
# partition_name
