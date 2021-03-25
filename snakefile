import os
import yaml

import pandas as pd

from snakemake_utils import generate_cv_data, list2str

cli_config = config.copy()
user_config_file = cli_config["config"]
CUDA_VISIBLE_DEVICES = cli_config["gpu"]

with open(user_config_file, 'r') as user_config:
    config = yaml.safe_load(user_config)

srcdir = os.path.join(workflow.basedir, "src")
scriptdir = os.path.join(workflow.basedir, "scripts")

# General data
output_dir = config["output_dir"]
work_dir = config["work_dir"]

# Tomogram lists
training_tomos = config["tomos_sets"]["training_list"]
prediction_tomos = config["tomos_sets"]["prediction_list"]

# Training
partition_name = "train_partition"
semantic_classes = config["training"]["semantic_classes"]

# prediction:
pred_class = config["prediction"]["semantic_class"]
pr_active = config["evaluation"]["particle_picking"]["active"]
pr_tolerance_radius = config["evaluation"]["particle_picking"]["pr_tolerance_radius"]

str_segmentation_names = list2str(my_list=semantic_classes)

if config["cluster"]["logdir"] is not None:
    os.makedirs(config["cluster"]["logdir"], exist_ok=True)

if config["cross_validation"]["active"]:
    model_name = os.path.basename(config["model_path"])[:-4] + "_{fold}"
    model_path = output_dir + "/models/" + model_name + ".pth"
else:
    model_path = config["model_path"]
    model_name = os.path.basename(config["model_path"])[:-4]

training_part_pattern = work_dir + "/training_data/{tomo_name}/.train_partition.{fold}.done"
done_training_pattern = ".done_patterns/" + model_path + "_{fold}.pth.done"
testing_part_pattern = work_dir + "/testing_data/{tomo_name}/partition.h5"
done_testing_part_pattern = work_dir + "/testing_data/{tomo_name}/.test_partition.{fold}.done"
segmented_part_pattern = ".done_patterns/" + model_name + ".{tomo_name}.{fold}.segmentation.done"
# noinspection PyTypeChecker
assemble_probability_map_done = output_dir + "/predictions/" + model_name + "/{tomo_name}/" + pred_class + \
                                "/.{fold}.probability_map.done"
# noinspection PyTypeChecker
postprocess_prediction_done = output_dir + "/predictions/" + model_name + "/{tomo_name}/" + pred_class + \
                              "/.{fold}.post_processed_prediction.mrc"
# noinspection PyTypeChecker
particle_picking_pr_done = output_dir + "/predictions/" + model_name + "/{tomo_name}/" + pred_class + \
                           "/pr_radius_" + str(pr_tolerance_radius) + "/detected/.{fold}.done_pp_snakemake"
# noinspection PyTypeChecker
dice_evaluation_done = output_dir + "/predictions/" + model_name + "/{tomo_name}/" + pred_class + \
                       "/.{fold}.done_dice_eval_snakemake"

if config["cross_validation"]["active"]:
    print("cross validation is active!")
    cv_folds = config["cross_validation"]["folds"]
    cv_data_path = output_dir + "/cv_data.csv"
    folds = list(range(cv_folds))

    assert len(training_tomos) >= cv_folds
    assert cv_folds >= 2

    if os.path.isfile(cv_data_path):
        print("cv partition exists")
    else:
        print("generating cv_data.csv file")
        generate_cv_data(cv_data_path=cv_data_path, tomo_training_list=training_tomos, cv_folds=cv_folds)

    cv_data = pd.read_csv(cv_data_path)
    cv_data.set_index("fold", inplace=True)

    #training_part_pattern = work_dir + "/training_data/{tomo_name}/fold_{fold}/partition.h5"
    #done_training_pattern = ".done_patterns/" + model_path + ".done"

    targets = []
    targets += expand([done_training_pattern], fold=folds)
    targets += expand([training_part_pattern], tomo_name=training_tomos, fold=folds)
    targets += expand([done_testing_part_pattern], tomo_name=training_tomos, fold=folds)
    targets += expand([postprocess_prediction_done], tomo_name=training_tomos, fold=folds)

    if config["evaluation"]["particle_picking"]["active"]:
        targets += expand([particle_picking_pr_done], tomo_name=training_tomos, fold=folds)
    else:
        os.makedirs(".done_patterns/"+ os.path.dirname(model_path), exist_ok=True)
        with open(".done_patterns/"+ model_path + ".skip_pr", mode="w") as f:
            print("skipping prediction")
    if config["evaluation"]["segmentation_evaluation"]["active"]:
        targets += expand([dice_evaluation_done], tomo_name=training_tomos, fold=folds)

else:
    folds = [None]

    print("training tomograms:", training_tomos)
    targets = []
    if config["training"]["active"]:
        print("training is active")
        targets += expand([done_training_pattern], fold=folds)

    if config["prediction"]["active"]:
        targets += expand([assemble_probability_map_done], tomo_name=prediction_tomos, fold=folds)

    if config["postprocessing_clustering"]["active"]:
        targets += expand([postprocess_prediction_done], tomo_name=prediction_tomos, fold=folds)
    else:
        os.makedirs(".done_patterns", exist_ok=True)
        with open(".done_patterns/.skip_prediction", mode="w") as f:
            print("skipping prediction")
        if config["postprocessing_clustering"]["active"]:
            print("postprocessing active")
            targets += expand([postprocess_prediction_done], tomo_name=prediction_tomos, fold=folds)

    if config["evaluation"]["particle_picking"]["active"]:
        targets += expand([particle_picking_pr_done], tomo_name=prediction_tomos, fold=folds)
    else:
        os.makedirs(".done_patterns", exist_ok=True)
        with open(".done_patterns/.skip_pr", mode="w") as f:
            print("skipping prediction")
    if config["evaluation"]["segmentation_evaluation"]["active"]:
        targets += expand([dice_evaluation_done], tomo_name=prediction_tomos, fold=folds)

if config["debug"]:
    print("TARGETS:\n")
    print(targets)

rule all:
    input:
         targets

rule partition_training:
    conda:
         "environment.yaml"
    output:
          done_file=training_part_pattern
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
         --config_file {user_config_file} \
         """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

rule training_3dunet:
    conda:
          "environment.yaml"
    input:
          training_set_done_file=expand([training_part_pattern], tomo_name=training_tomos, fold=folds)
    output:
          done=[done_training_pattern,model_path] if config["cross_validation"]["active"] else done_training_pattern
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="2-05:00:00",
          nodes=1,
          cores=4,
          memory="50G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:4'
    resources:
             gpu=4
    shell:
         f"""
        python3 {scriptdir}/training.py \
        --pythonpath {srcdir} \
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --gpu $CUDA_VISIBLE_DEVICES"

rule predict_partition:
    conda:
         "environment.yaml"
    output:
          done_file=done_testing_part_pattern
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
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

rule segment:
    conda:
         "environment.yaml"
    input:
         done=[done_training_pattern, done_testing_part_pattern] if
         config["training"]["active"] or config["cross_validation"]["active"] else
         done_testing_part_pattern
    output:
          file=segmented_part_pattern
    params:
          config=user_config_file,
          logdir=config["cluster"]["logdir"],
          walltime="01:20:00",
          nodes=1,
          cores=4,
          memory="80G",
          gres='#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    resources:
             gpu=2
    shell:
         f"""
        python3 {scriptdir}/segment.py \
        --pythonpath {srcdir} \
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name} \
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
          memory="20G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/assemble_prediction.py \
        --pythonpath {srcdir} \
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

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
          walltime="08:30:00",
          nodes=1,
          cores=4,
          memory="30G",
          gres=''
    shell:
         f"""
        python3 {scriptdir}/clustering_and_cleaning.py \
        --pythonpath {srcdir} \
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

rule particle_picking_evaluation:
    conda:
         "environment.yaml"
    input:
         postprocess_prediction_done
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
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

rule segmentation_evaluation:
    conda:
         "environment.yaml"
    input:
         file=postprocess_prediction_done
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
        --config_file {user_config_file} \
        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"

#rule filter:
#    conda:
#         "environment.yaml"
#    output:
#          file=filtered_file_done
#    params:
#          config=user_config_file,
#          logdir=config["cluster"]["logdir"],
#          walltime="00:20:00",
#          nodes=1,
#          cores=2,
#          memory="15G",
#          gres=''
#    shell:
#         f"""
#        python3 {scriptdir}/match_spectrum.py \
#        --pythonpath {srcdir} \
#        --config_file {user_config_file} \
#        """ + "--fold {wildcards.fold} --tomo_name {wildcards.tomo_name}"
