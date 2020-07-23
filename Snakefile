import os, yaml
import pandas as pd
from snakemake_utils import generate_cv_folds

cli_config = config.copy()
user_config_file = cli_config["config"]
CUDA_VISIBLE_DEVICES = cli_config["gpu"]

with open(user_config_file, 'r') as user_config:
    config = yaml.safe_load(user_config)

srcdir = os.path.join(workflow.basedir, "src")
scriptdir = os.path.join(workflow.basedir, "scripts")


def generate_training_list_str(training_tomos: list) -> str:
    tomo_training_list = ""
    for tomo in training_tomos:
        tomo_training_list += tomo + " "
    return tomo_training_list


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

str_segmentation_names = ""
for name in segmentation_names:
    str_segmentation_names += name + " "

tomo_training_list = generate_training_list_str(training_tomos=training_tomos)

if config["cross_validation"]["active"]:
    model_path = os.path.join(output_dir, "models")
    model_path = os.path.join(model_path, model_name)
else:
    model_path = model_name

training_part_pattern = work_dir + "/training_data/{tomo_name}/train_partition.h5"
training_model_pattern = model_path
done_training_pattern = ".done_patterns/" + model_path + ".done"
testing_part_pattern = work_dir + "/test_partitions/{tomo_name}/" + model_name[:-4] + "/test_partition.h5"
segmented_part_pattern = ".done_patterns/" + model_path + ".{tomo_name}.segmentation.done"
assemble_probability_map_done = output_dir + "/predictions/" + model_name[
                                                               :-4] + "/{tomo_name}/" + pred_class + "/probability_map.mrc"
postprocess_prediction_done = output_dir + "/predictions/" + model_name[
                                                             :-4] + "/{tomo_name}/" + pred_class + "/post_processed_prediction.mrc"
particle_picking_pr_done = output_dir + "/predictions/" + model_name[
                                                          :-4] + "/{tomo_name}/" + pred_class + "/pr_radius_" + str(
    pr_tolerance_radius) + "/detected/.done_pp_snakemake"
dice_evaluation_done = output_dir + "/predictions/" + model_name[
                                                      :-4] + "/{tomo_name}/" + pred_class + "/.done_dice_eval_snakemake"
if config["cluster"]["logdir"] is not None:
    os.makedirs(config["cluster"]["logdir"], exist_ok=True)

print("training_tomos", training_tomos)
targets = []
if config["training"]["active"]:
    print("training is active")
    targets.append(model_path)
    targets.append(done_training_pattern)

if config["prediction"]["active"]:
    targets += expand([postprocess_prediction_done], tomo_name=prediction_tomos)
else:
    with open(".done_patterns/.skip_prediction", mode="w") as f:
        print("skipping prediction")

if config["evaluation"]["particle_picking"]["active"]:
    targets += expand([particle_picking_pr_done], tomo_name=prediction_tomos)

if config["evaluation"]["segmentation_evaluation"]["active"]:
    targets += expand([dice_evaluation_done], tomo_name=prediction_tomos)

if config["cross_validation"]["active"]:
    cv_folds = config["cross_validation"]["cv_folds"]
    cv_statistics_file = "dice_" + config["cross_validation"]["statistics_file"]
    cv_statistics_file_pp = "pp_" + config["cross_validation"]["statistics_file"]
    print("todo assert len(training_tomos) > cv_folds", len(training_tomos), cv_folds)
    print("tomo assert cv_folds > 1")
    assert len(training_tomos) >= cv_folds
    assert cv_folds >= 2
    if config["cross_validation"]["cv_data"] is not None:
        path_to_cv_data = config["cross_validation"]["cv_data"]
    else:
        print("generating cv_data.csv file")
        path_to_cv_data = "cv_data.csv"
        cv_training_dict, cv_evaluation_dict = generate_cv_folds(tomo_training_list=training_tomos, cv_folds=cv_folds)
        cv_data = pd.DataFrame({"cv_fold": list(cv_training_dict.keys()),
                                "cv_training_list": [None for _ in range(cv_folds)],
                                "cv_evaluation_list": [None for _ in range(cv_folds)]})
        cv_data.set_index("cv_fold", inplace=True)
        for fold in cv_training_dict.keys():
            print("fold:", fold)
            cv_training_tomos = cv_training_dict[fold]
            cv_evaluation_tomos = cv_evaluation_dict[fold]
            cv_data["cv_training_list"][fold] = generate_training_list_str(training_tomos=cv_training_tomos)
            cv_data["cv_evaluation_list"][fold] = generate_training_list_str(training_tomos=cv_evaluation_tomos)

        cv_data_path = cv_data.to_csv(path_to_cv_data)
        del cv_data, cv_evaluation_dict, cv_training_tomos

    targets = []
    done_cv_eval = ".done_patterns/" + model_path[:-4] + "_{fold}.pkl.done"
    done_cv_pp = ".done_patterns/" + model_path[:-4] + "_{fold}.pkl.done_pp_cv_snakemake"
    targets += expand([done_cv_pp], fold=list(range(cv_folds)))
print("TARGETS:\n")
print(targets)

rule all:
    input:
         targets

rule partition_training:
    conda:
         "environment.yaml"
    output:
          file=training_part_pattern
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
         """ + "--tomo_name {wildcards.tomo_name}"

rule training_3dunet:
    conda:
         "environment.yaml"
    input:
         training_set_done_file=expand([training_part_pattern], tomo_name=training_tomos)
    output:
          file=model_path,
          done=done_training_pattern
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="04:30:00",
          nodes=1,
          cores=4,
          memory="40G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    resources:
             gpu=2
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
        """ + "--gpu $CUDA_VISIBLE_DEVICES"

rule cross_validation_3dunet:
    conda:
         "environment.yaml"
    input:
         training_set_done_file=expand([training_part_pattern], tomo_name=training_tomos)
    output:
          done=".done_patterns/" + model_path[:-4] + "_{fold}.pkl.done"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="10:00:00",
          nodes=1,
          cores=4,
          memory="40G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:4'
    resources:
             gpu=4
    shell:
         f"""
         python3 {scriptdir}/cross_validation_unet.py \
         --pythonpath {srcdir} \
         --partition_name  {{partition_name}} \
         --segmentation_names  {{segmentation_names}} \
         --dataset_table  {{dataset_table}} \
         --output_dir  {{output_dir}} \
         --work_dir  {{work_dir}} \
         --box_shape  {{box_shape}} \
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
         --statistics_file {{cv_statistics_file}} \
         --cv_data_path {{path_to_cv_data}} \
         --fold {{wildcards.fold}} \
         """ + "--gpu $CUDA_VISIBLE_DEVICES"

rule predict_partition:
    conda:
         "environment.yaml"
    output:
          file=testing_part_pattern
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:40:00",
          nodes=1,
          cores=1,
          memory="30G",
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
        --box_shape  {{box_shape}} \
        --overlap  {{overlap}} \
        """ + "--tomo_name {wildcards.tomo_name}"

rule segment:
    conda:
         "environment.yaml"
    input:
         done=[model_path, done_training_pattern, testing_part_pattern] if config["training"]["active"] else
         testing_part_pattern
    output:
          file=segmented_part_pattern
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="00:30:00",
          nodes=1,
          cores=4,
          memory="20G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    resources:
             gpu=2
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
        --depth  {{depth}} \
        --decoder_dropout  {{decoder_dropout}} \
        --encoder_dropout  {{encoder_dropout}} \
        --batch_size  {{batch_size}} \
        --batch_norm  {{BatchNorm}} \
        --initial_features  {{initial_features}} \
        --segmentation_names  {{segmentation_names}} \
        --box_shape  {{box_shape}} \
        --overlap  {{overlap}} \
        """ + "--tomo_name {wildcards.tomo_name} \
         --gpu $CUDA_VISIBLE_DEVICES"

rule assemble_prediction:
    conda:
         "environment.yaml"
    input:
         segmented_part_pattern
    output:
          file=assemble_probability_map_done
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
        --segmentation_names  {{segmentation_names}} \
        --box_shape  {{box_shape}} \
        --overlap  {{overlap}} \
        """ + "--tomo_name {wildcards.tomo_name}"

rule postprocess_prediction:
    conda:
         "environment.yaml"
    input:
         assemble_probability_map_done
    output:
          postprocess_prediction_done
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="01:30:00",
          nodes=1,
          cores=4,
          memory="30G",
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
        --filtering_mask {{filtering_mask}} \
        --segmentation_names  {{segmentation_names}} \
        """ + "--tomo_name {wildcards.tomo_name}"

rule particle_picking_evaluation:
    conda:
         "environment.yaml"
    input:
          postprocess_prediction_done if config["prediction"]["active"] else ".done_patterns/.skip_prediction"
    output:
          file=particle_picking_pr_done
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
        --statistics_file {{pr_statistics_file}} \
        --radius {{pr_tolerance_radius}} \
        --segmentation_names  {{segmentation_names}} \
        """ + "--tomo_name {wildcards.tomo_name}"

rule segmentation_evaluation:
    conda:
         "environment.yaml"
    input:
          postprocess_prediction_done if config["prediction"]["active"] else ".done_patterns/.skip_prediction"
    output:
          file=dice_evaluation_done
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
        --segmentation_names  {{segmentation_names}} \
        """ + "--tomo_name {wildcards.tomo_name}"

rule cross_validation_particle_picking:
    conda:
         "environment.yaml"
    input:
         done_cv_eval=".done_patterns/" + model_path[:-4] + "_{fold}.pkl.done",
    output:
          done=".done_patterns/" + model_path[:-4] + "_{fold}.pkl.done_pp_cv_snakemake"
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="05:00:00",
          nodes=1,
          cores=4,
          memory="40G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:4'
    resources:
             gpu=4
    shell:
         f"""
         python3 {scriptdir}/cross_validation_particle_picking.py \
         --pythonpath {srcdir} \
         --test_partition  {{pred_partition_name}} \
         --dataset_table  {{dataset_table}} \
         --output_dir  {{output_dir}} \
         --work_dir  {{work_dir}} \
         --processing_tomo  {{pred_processing_tomo}} \
         --model_name {{model_name}} \
         --statistics_file {{cv_statistics_file_pp}} \
         --cv_data_path {{path_to_cv_data}} \
         --fold {{wildcards.fold}} \
         --class_number {{pred_class_number}} \
         --min_cluster_size {{min_cluster_size}} \
         --max_cluster_size {{max_cluster_size}} \
         --threshold {{threshold}} \
         --dataset_table {{dataset_table}} \
         --filtering_mask {{filtering_mask}} \
         --calculate_motl {{calculate_motl}} \
         --radius {{pr_tolerance_radius}} \
         """ + "--gpu $CUDA_VISIBLE_DEVICES"
