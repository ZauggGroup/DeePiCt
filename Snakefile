"""
Modular pipeline for Spatial Point Pattern Analysis in 3D
"""

__author__ = "Irene de Teresa"
__version__ = "0.1.0"
__maintainer__ = "Irene de Teresa"
__email__ = "irene.de.teresa@embl.de"

import os, yaml

# from datetime import datetime

# Load user's config file
cli_config = config.copy()
user_config_file = cli_config["config"]
CUDA_VISIBLE_DEVICES = cli_config["gpu"]

# if CUDA_VISIBLE_DEVICES == '':
#     CUDA_VISIBLE_DEVICES='0'
# print("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)
with open(user_config_file, 'r') as user_config:
    config = yaml.safe_load(user_config)

srcdir = os.path.join(workflow.basedir, "src")
scriptdir = os.path.join(workflow.basedir, "scripts")

# run_name = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")

# General data
dataset_table = config["dataset_table"]
output_dir = config["output_dir"]
work_dir = config["work_dir"]
model_name = config["model_name"]

# Tomogram lists
training_tomos = config["tomos_sets"]["training_list"]
prediction_tomos = config["tomos_sets"]["prediction_list"]
# Training
overlap = config["training"]["overlap"]
partition_name = "train_partition"
segmentation_names = config["training"]["semantic_classes"]
processing_tomo = config["training"]["processing_tomo"]
box_shape = config["training"]["box_shape"]
min_label_fraction = config["training"]["min_label_fraction"]
max_label_fraction = 1

#unet_hyperparameters:
depth = config["training"]["unet_hyperparameters"]["depth"]
initial_features = config["training"]["unet_hyperparameters"]["initial_features"]
n_epochs = config["training"]["unet_hyperparameters"]["n_epochs"]
split = config["training"]["unet_hyperparameters"]["split"]
BatchNorm = config["training"]["unet_hyperparameters"]["BatchNorm"]
encoder_dropout = config["training"]["unet_hyperparameters"]["encoder_dropout"]
decoder_dropout = config["training"]["unet_hyperparameters"]["decoder_dropout"]
batch_size = config["training"]["unet_hyperparameters"]["batch_size"]

# prediction:
pred_processing_tomo = config["prediction"]["processing_tomo"]
pred_partition_name = "test_partition"
pred_class = config["prediction"]["semantic_class"]
pred_class_number = -1
for class_number, semantic_class in enumerate(segmentation_names):
    if semantic_class == pred_class:
        pred_class_number = class_number
assert pred_class_number >= 0, "Prediction class not among segmentation names for this model!"

# Thresholding clustering and motl generation
threshold = config["postprocessing_clustering"]["threshold"]
min_cluster_size = config["postprocessing_clustering"]["min_cluster_size"]
max_cluster_size = config["postprocessing_clustering"]["max_cluster_size"]
calculate_motl = config["postprocessing_clustering"]["calculate_motl"]
ignore_border_thickness = config["postprocessing_clustering"]["ignore_border_thickness"]
filtering_mask = config["postprocessing_clustering"]["filtering_mask"]

# evaluation:
# a. For precision recall in particle picking
pr_active = config["evaluation"]["particle_picking"]["active"]
pr_tolerance_radius = config["evaluation"]["particle_picking"]["pr_tolerance_radius"]
pr_statistics_file = config["evaluation"]["particle_picking"]["statistics_file"]

# b. For dice coefficient evaluation at the voxel level
dice_eval_active = config["evaluation"]["segmentation_evaluation"]["active"]
dice_eval_statistics_file = config["evaluation"]["segmentation_evaluation"]["statistics_file"]

os.makedirs(".done_patterns", exist_ok=True)
done_training_set_generation_pattern = expand([".done_patterns/run_3d_training.done_set_generation_{tomo}.done"],
                                              tomo=training_tomos)
done_training_pattern = [".done_patterns/run_3d_training.done"]

done_prediction_set_generation_pattern = expand(".done_patterns/run_prediction_set_generation_{tomo}.done",
                                                tomo=prediction_tomos)
done_segmentation_pattern = expand(".done_patterns/run_segmentation_{tomo}.done", tomo=prediction_tomos)
done_assembling_prediction_pattern = expand([".done_patterns/run_assembling_prediction_{tomo}.done"],
                                            tomo=prediction_tomos)
done_clustering_pattern = expand(".done_patterns/run_clustering_{tomo}.done", tomo=prediction_tomos)
done_particle_picking_evaluation_pattern = expand([".done_patterns/run_particle_picking_evaluation_{tomo}.done"],
                                                  tomo=prediction_tomos)
done_segmentation_evaluation_pattern = expand([".done_patterns/run_dice_evaluation_{tomo}.done"],
                                              tomo=prediction_tomos)
skip_training_pattern = ".done_patterns/run_skip_training"
skip_prediction_pattern = ".done_patterns/run_skip_prediction"
skip_particle_picking_evaluation_pattern = ".done_patterns/{}_skip_particle_picking_eval"

if config["prediction"]["active"]:
    if config["evaluation"]["particle_picking"]["active"]:
        start_segmentation_evaluation_when = ".done_patterns/run_particle_picking_evaluation_{tomo}.done"
    else:
        start_segmentation_evaluation_when = ".done_patterns/run_clustering_{tomo}.done"
else:
    if config["evaluation"]["particle_picking"]["active"]:
        start_segmentation_evaluation_when = ".done_patterns/run_particle_picking_evaluation_{tomo}.done"
    else:
        start_segmentation_evaluation_when = ".done_patterns/run_skip_prediction"

str_segmentation_names = ""
for name in segmentation_names:
    str_segmentation_names += name + " "

tomo_training_list = ""
for tomo in training_tomos:
    tomo_training_list += tomo + " "

model_path = os.path.join(output_dir, "models")
model_path = os.path.join(model_path, model_name)

# Build dependency targets:
targets = []
if config["training"]["active"]:
    targets += [done_training_pattern]
else:
    print("training isn't active!")
    with open(file=skip_training_pattern, mode="w"):
        print(skip_training_pattern)

if config["prediction"]["active"]:
    targets += done_prediction_set_generation_pattern
    targets += done_segmentation_pattern
    targets += done_assembling_prediction_pattern
    targets += done_clustering_pattern
else:
    print("prediction isn't active!")
    print(skip_prediction_pattern)
    with open(file=skip_prediction_pattern, mode="w"):
        print(skip_prediction_pattern)

if config["evaluation"]["particle_picking"]["active"]:
    targets += done_particle_picking_evaluation_pattern
if config["evaluation"]["segmentation_evaluation"]["active"]:
    targets += done_segmentation_evaluation_pattern

if config["debug"]:
    print("TARGETS:\n", targets)
    final_line = "rm -r .done_patterns/*"
else:
    final_line = "echo 'Finishing pipeline.'"

if config["cluster"]["logdir"]:
    os.makedirs(config["cluster"]["logdir"], exist_ok=True)

# Rules
rule all:
    input:
         targets
    shell:
         final_line

rule training_set_generation:
    conda:
         "environment.yaml"
    output:
          file=".done_patterns/run_3d_training.done_set_generation_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="01:10:00",
          nodes=1,
          cores=1,
          memory="30G",
          gres=''
    shell:
         f"""
         python3 {scriptdir}/generate_training_data.py \
         --pythonpath {srcdir} \
         --overlap {{overlap}}  \
         --partition_name {{partition_name}} \
         --segmentation_names {{str_segmentation_names}} \
         --dataset_table {{dataset_table}} \
         --output_dir {{output_dir}} \
         --work_dir {{work_dir}} \
         --processing_tomo {{processing_tomo}} \
         --box_shape {{box_shape}} \
         --min_label_fraction {{min_label_fraction}} \
         --max_label_fraction {{max_label_fraction}} \
         """ + "--tomo_name {wildcards.tomo} \
                 && touch {output.file}"
# "echo {output.file} \
#  && touch {output.file}"

rule training_3dunet:
    conda:
         "environment.yaml"
    input:
         training_set_done_file=done_training_set_generation_pattern
    output:
          file=".done_patterns/run_3d_training.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="8:00:00",
          nodes=1,
          cores=4,
          memory="40G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    resources:
             gpu=1
    shell:
         f"""
        python3 {scriptdir}/training.py \
        --pythonpath {srcdir} \
        --partition_name  {{partition_name}} \
        --segmentation_names  {{segmentation_names}} \
        --dataset_table  {{dataset_table}} \
        --output_dir  {{output_dir}} \
        --work_dir  {{work_dir}} \
        --box_shape  {{box_shape}} \
        --output_dir  {{output_dir}} \
        --tomo_training_list  {{tomo_training_list}} \
        --split  {{split}} \
        --n_epochs  {{n_epochs}} \
        --depth  {{depth}} \
        --decoder_dropout  {{decoder_dropout}} \
        --encoder_dropout  {{encoder_dropout}} \
        --batch_size  {{batch_size}} \
        --batch_norm  {{BatchNorm}} \
        --initial_features  {{initial_features}} \
        --overlap  {{overlap}} \
        --processing_tomo  {{processing_tomo}} \
        --partition_name  {{partition_name}} \
        --model_name {{model_name}} \
        """ + " --gpu $CUDA_VISIBLE_DEVICES \
        && touch {output.file}"

rule generate_prediction_partition:
    conda:
         "environment.yaml"
    input:
         ".done_patterns/run_3d_training.done" if config["training"][
             "active"] else ".done_patterns/run_skip_training"
    output:
          file=".done_patterns/run_prediction_set_generation_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="8:00:00",
          nodes=1,
          cores=1,
          memory="10G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/generate_prediction_partition.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --work_dir {{work_dir}} \
        --model_name {{model_name}} \
        --test_partition {{pred_partition_name}} \
        --processing_tomo {{pred_processing_tomo}} \
        """ + "--tomo_name {wildcards.tomo} \
        && touch {output.file}"

rule segment:
    conda:
         "environment.yaml"
    input:
         ".done_patterns/run_prediction_set_generation_{tomo}.done"
    output:
          file=".done_patterns/run_segmentation_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:20:00",
          nodes=1,
          cores=4,
          memory="10G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    resources:
             gpu=1
    shell:
         f"""
        python3 {scriptdir}/segment.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --work_dir {{work_dir}} \
        --model_name {{model_name}} \
        --test_partition {{pred_partition_name}} \
        --processing_tomo {{pred_processing_tomo}} \
        """ + "--tomo_name {wildcards.tomo} \
         --gpu $CUDA_VISIBLE_DEVICES \
        && touch {output.file}"

rule assemble_prediction:
    conda:
         "environment.yaml"
    input:
         ".done_patterns/run_segmentation_{tomo}.done"
    output:
          file=".done_patterns/run_assembling_prediction_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:20:00",
          nodes=1,
          cores=2,
          memory="10G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    shell:
         f"""
        python3 {scriptdir}/assemble_prediction.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --work_dir {{work_dir}} \
        --model_name {{model_name}} \
        --test_partition {{pred_partition_name}} \
        --class_number {{pred_class_number}} \
        --processing_tomo {{pred_processing_tomo}} \
        """ + "--tomo_name {wildcards.tomo} \
        && touch {output.file}"

rule clustering_and_cleaning:
    conda:
         "environment.yaml"
    input:
         ".done_patterns/run_assembling_prediction_{tomo}.done"
    output:
          file=".done_patterns/run_clustering_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:40:00",
          nodes=1,
          cores=4,
          memory="10G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/clustering_and_cleaning.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --model_name {{model_name}} \
        --class_number {{pred_class_number}} \
        --min_cluster_size {{min_cluster_size}} \
        --max_cluster_size {{max_cluster_size}} \
        --threshold {{threshold}} \
        --dataset_table {{dataset_table}} \
        --calculate_motl {{calculate_motl}} \
        """ + "--tomo_name {wildcards.tomo} \
        && touch {output.file}"

rule particle_picking_evaluation:
    conda:
         "environment.yaml"
    input:
         done_clustering_and_cleaning_file=".done_patterns/run_clustering_{tomo}.done" if config["prediction"][
             "active"] else skip_prediction_pattern
    output:
          file=".done_patterns/run_particle_picking_evaluation_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:10:00",
          nodes=1,
          cores=2,
          memory="10G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/particle_picking_evaluation.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --model_name {{model_name}} \
        --class_number {{pred_class_number}} \
        --calculate_motl {{calculate_motl}} \
        --filtering_mask {{filtering_mask}} \
        --calculate_motl {{calculate_motl}} \
        --statistics_file {{pr_statistics_file}} \
        --radius {{pr_tolerance_radius}} \
        """ + "--tomo_name {wildcards.tomo} \
        && touch {output.file}"

rule segmentation_evaluation:
    conda:
         "environment.yaml"
    input:
         start_segmentation_evaluation_when
    output:
          file=".done_patterns/run_dice_evaluation_{tomo}.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:10:00",
          nodes=1,
          cores=2,
          memory="15G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/segmentation_evaluation.py \
        --pythonpath {srcdir} \
        --dataset_table {{dataset_table}} \
        --output_dir {{output_dir}} \
        --model_name {{model_name}} \
        --class_number {{pred_class_number}} \
        --filtering_mask {{filtering_mask}} \
        --statistics_file {{dice_eval_statistics_file}} \
        """ + "--tomo_name {wildcards.tomo} \
        && touch {output.file}"
