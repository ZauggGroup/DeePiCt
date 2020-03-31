import os, yaml
import os.path as op
import pandas as pd
from datetime import datetime


# Load user config ffile
cli_config = config.copy()
user_config_file = cli_config["config"]
with open(user_config_file, 'r') as user_config:
    config = yaml.safe_load(user_config)

srcdir = workflow.basedir

# Workaround so filter rule can be applied to both training and pred tomos
filter_meta = pd.DataFrame({"prefix":[], "data":[]}) 

# Training data management
if (
    config["training"]["evaluation"]["active"] or
    config["training"]["production"]["active"] or
    (config["prediction"]["active"] and config["prediction"]["model"] is None)
):
    training_meta = pd.read_csv(config["data"]["training_data"])
    
    if config["data"]["train_workdir"]:
        os.makedirs(config["data"]["train_workdir"], exist_ok=True)
        training_meta["prefix"] = config["data"]["train_workdir"] + '/' + training_meta["data"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    else:
        training_meta["prefix"] = training_meta["data"].apply(lambda x: os.path.splitext(x)[0])
    
    training_meta["filtered"] = training_meta["prefix"] + "_filtered.mrc"
    training_meta["labels_remapped"] = training_meta["prefix"] + "_labels_remapped.mrc"
    training_meta["slices"] = training_meta["prefix"] + "_slices.h5"

    filter_meta = filter_meta.merge(training_meta[["data", "prefix"]], how="outer")

    training_meta.set_index("prefix", inplace=True)


    all_training_slices = training_meta["slices"].to_list()
else:
    all_training_slices = []


# Prediction data management
if config["prediction"]["active"] | config["postprocessing"]["active"]:
    prediction_meta = pd.read_csv(config["data"]["prediction_data"])

    if config["data"]["output_dir"]:
        os.makedirs(config["data"]["output_dir"], exist_ok=True)
        prediction_meta["prefix"] = config["data"]["output_dir"] + '/' + prediction_meta["data"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    else:
        prediction_meta["prefix"] = prediction_meta["data"].apply(lambda x: os.path.splitext(x)[0])
    
    prediction_meta["filtered"] = prediction_meta["prefix"] + "_filtered.mrc"
    prediction_meta["prediction"] = prediction_meta["prefix"] + "_pred.mrc"
    prediction_meta["polished"] = prediction_meta["prefix"] + "_pred_polished.mrc"

    filter_meta = filter_meta.merge(prediction_meta[["data", "prefix"]], how="outer")

    prediction_meta.set_index("prefix", inplace=True)

filter_meta.set_index("prefix", inplace=True)

# Pipeline targets
targets = []
timestamp = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")
eval_done_file = '.snakemake/' + timestamp + "_eval.done"

if config["training"]["evaluation"]["active"]:
    targets.append(eval_done_file)

if config["training"]["production"]["active"]:
    targets.append(config["training"]["production"]["model_output"])

if config["prediction"]["active"]:
    targets += prediction_meta["prediction"].to_list()

if config["postprocessing"]["active"]:
    targets += prediction_meta["polished"].to_list()

if config.get("debug"):
    print("TARGETS:\n", targets)
    print("TRAINING_META:\n", training_meta)
    print("PREDICTION_META:\n", prediction_meta)

# Intermediate file patterns
filtered_pattern = "{prefix}_filtered.mrc"
remapped_labels_pattern = "{prefix}_labels_remapped.mrc"
slices_pattern = "{prefix}_slices.h5"
prediction_pattern = "{prefix}_pred.mrc"
postprocessed_pattern = "{prefix}_pred_polished.mrc"



# Rules
rule all:
    input: 
        targets

rule filter_tomogram:
    input:
        tomo = lambda wildcards: filter_meta.loc[wildcards.prefix, "data"] # This approach allows to use both .mrc and .rec
    output:
        filtered_tomo = filtered_pattern
    params:
        lowpass_cutoff  = config["preprocessing"]["filtering"]["lowpass_cutoff"],
        highpass_cutoff = config["preprocessing"]["filtering"]["highpass_cutoff"],
        clamp_nsigma    = config["preprocessing"]["filtering"]["clamp_nsigma"],
        logdir      = config["cluster"]["logdir"],
        walltime    = "0:30:00",
        nodes       = 1,
        cores       = 8,
        memory      = "16G",
        gres        = ''
    shell:
        """
        module load EMAN2
        e2proc3d.py {input.tomo} {output.filtered_tomo} \
        --process filter.lowpass.gauss:cutoff_abs={params.lowpass_cutoff} \
        --process filter.highpass.gauss:cutoff_pixels={params.highpass_cutoff} \
        --process normalize \
        --process threshold.clampminmax.nsigma:nsigma={params.clamp_nsigma} \
        """

rule remap_labels:
    input:
        labels = lambda wildcards: training_meta.loc[wildcards.prefix, "labels"],
    params:
        mapping = config["preprocessing"]["remapping"]["mapping"]
    conda:
        "envs/keras-env.yaml"
    output:
        remapped_labels = remapped_labels_pattern,
    params:
        logdir      = config["cluster"]["logdir"],
        walltime    = "0:10:00",
        nodes       = 1,
        cores       = 2,
        memory      = "16G",
        gres        = ''
    shell:
        f"""
        python3 {srcdir}/scripts/remap_labels.py \
        --input {{input.labels}} \
        --output {{output.remapped_labels}} \
        --mapping {{params.mapping}}
        """

rule slice_tomogram:
    input:
        tomo = filtered_pattern if config["preprocessing"]["filtering"]["active"] else lambda wildcards: training_meta.loc[wildcards.prefix, "data"],
        labels = (lambda wildcards: training_meta.loc[wildcards.prefix, "labels_remapped"]) if config["preprocessing"]["remapping"]["active"] else (lambda wildcards: training_meta.loc[wildcards.prefix, "labels"])
        # This way the names can differ between tomo and labels
    output:
        sliced_tomo = slices_pattern
    conda:
        "envs/keras-env.yaml"
    params:
        config = user_config_file,
        flip_y = lambda wildcards: training_meta.loc[wildcards.prefix, "flip_y"] * "--flip_y",
        logdir      = config["cluster"]["logdir"],
        walltime    = "0:10:00",
        nodes       = 1,
        cores       = 2,
        memory      = "16G",
        gres        = ''
    shell:
        f"""
        python3 {srcdir}/scripts/create_training_data.py \
        --features {{input.tomo}} \
        --labels {{input.labels}} \
        --output {{output.sliced_tomo}} \
        --config {{params.config}} \
        {{params.flip_y}} \
        """

rule train_evaluation_model:
    input:
        training_data = all_training_slices
    output:
        eval_done_file = eval_done_file
    conda:
        "envs/keras-env.yaml"
    params:
        config      = user_config_file,
        logdir      = config["cluster"]["logdir"],
        walltime    = "8:00:00",
        nodes       = 1,
        cores       = 4,
        memory      = "32G",
        gres        = '#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    shell:
        f"""
        python3 {srcdir}/scripts/train_eval_model.py \
        --config {{params.config}} \
        --datasets {{input.training_data}} \
        && touch {{output.eval_done_file}} \
        """

rule train_production_model:
    input:
        training_data = all_training_slices
    output:
        model = config["training"]["production"]["model_output"] if config["training"]["production"]["active"] else []
    conda:
        "envs/keras-env.yaml"
    params:
        config      = user_config_file,
        logdir      = config["cluster"]["logdir"],
        walltime    = "2:00:00",
        nodes       = 1,
        cores       = 4,
        memory      = "32G",
        gres        = '#SBATCH -p gpu\n#SBATCH --gres=gpu:2'
    shell:
        f"""
        python3 {srcdir}/scripts/train_prod_model.py \
        --config {{params.config}} \
        --datasets {{input.training_data}} \
        """

rule predict_organelles:
    input:
        tomo = filtered_pattern if config["preprocessing"]["filtering"]["active"] else lambda wildcards: prediction_meta.loc[wildcards.prefix, "data"],
        model = config["training"]["production"]["model_output"] if config["prediction"]["model"] is None else config["prediction"]["model"]
    output:
        prediction = prediction_pattern
    conda:
        "envs/keras-env.yaml"
    params:
        config      = user_config_file,
        logdir      = config["cluster"]["logdir"],
        walltime    = "0:30:00",
        nodes       = 1,
        cores       = 4,
        memory      = "16G",
        gres        = '#SBATCH -p gpu\n#SBATCH --gres=gpu:1'
    shell:
        f"""
        python3 {srcdir}/scripts/predict_organelles.py \
        --features {{input.tomo}} \
        --output {{output.prediction}} \
        --model {{input.model}} \
        --config {{params.config}} \
        """

rule postprocess_organelles:
    input:
        pred = prediction_pattern
    output:
        polished_pred = postprocessed_pattern
    conda:
        "envs/keras-env.yaml"
    params:
        config      = user_config_file,
        logdir      = config["cluster"]["logdir"],
        walltime    = "0:10:00",
        nodes       = 1,
        cores       = 4,
        memory      = "8G",
        gres        = ''
    shell:
        f"""
        python3 {srcdir}/scripts/postprocess.py \
        --input {{input.pred}} \
        --output {{output.polished_pred}} \
        --config {{params.config}} \
        """
