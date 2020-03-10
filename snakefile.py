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

# Training data management
if (
    config["training"]["evaluation"]["active"] or
    config["training"]["production"]["active"] or
    (config["prediction"]["active"] and config["prediction"]["model"] is None)
):
    training_meta = pd.read_csv(config["data"]["training_data"])
    
    if config["data"]["output_dir"]:
        training_meta["prefix"] = config["data"]["output_dir"] + '/' + training_meta["data"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    else:
        training_meta["prefix"] = training_meta["data"].apply(lambda x: os.path.splitext(x)[0])
    
    training_meta["filtered"] = training_meta["prefix"] + "_filtered.mrc"
    training_meta["slices"] = training_meta["prefix"] + "_slices.h5"
    training_meta.set_index("prefix", inplace=True)

    all_training_slices = training_meta["slices"].to_list()
else:
    all_training_slices = []


# Prediction data management
if config["prediction"]["active"]:
    prediction_meta = pd.read_csv(config["data"]["prediction_data"])

    if config["data"]["output_dir"]:
        prediction_meta["prefix"] = config["data"]["output_dir"] + '/' + prediction_meta["data"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    else:
        prediction_meta["prefix"] = prediction_meta["data"].apply(lambda x: os.path.splitext(x)[0])
    
    prediction_meta["filtered"] = prediction_meta["prefix"] + "_filtered.mrc"
    prediction_meta["prediction"] = prediction_meta["prefix"] + "_pred.mrc"
    prediction_meta.set_index("prefix", inplace=True)


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


# Intermediate file patterns
filtered_pattern = "{prefix}_filtered.mrc"
slices_pattern = "{prefix}_slices.h5"
prediction_pattern = "{prefix}_pred.mrc"


# Rules
rule all:
    input: 
        targets

rule filter_training_tomogram:
    input:
        tomo = lambda wildcards: training_meta.loc[wildcards.prefix, "data"] # This approach allows to use both .mrc and .rec
    output:
        filtered_tomo = filtered_pattern
    params:
        lowpass_cutoff  = config["preprocessing"]["filtering"]["lowpass_cutoff"],
        highpass_cutoff = config["preprocessing"]["filtering"]["highpass_cutoff"],
        clamp_nsigma    = config["preprocessing"]["filtering"]["clamp_nsigma"]
    shell:
        """
        e2proc3d.py {input.tomo} {output.filtered_tomo} \
        --process filter.lowpass.gauss:cutoff_abs={params.lowpass_cutoff} \
        --process filter.highpass.gauss:cutoff_pixels={params.highpass_cutoff} \
        --process normalize \
        --process threshold.clampminmax.nsigma:nsigma={params.clamp_nsigma} \
        """

rule filter_prediction_tomogram:
    input:
        tomo = lambda wildcards: prediction_meta.loc[wildcards.prefix, "data"] # This approach allows to use both .mrc and .rec
    output:
        filtered_tomo = filtered_pattern
    params:
        lowpass_cutoff  = config["preprocessing"]["filtering"]["lowpass_cutoff"],
        highpass_cutoff = config["preprocessing"]["filtering"]["highpass_cutoff"],
        clamp_nsigma    = config["preprocessing"]["filtering"]["clamp_nsigma"]
    shell:
        """
        e2proc3d.py {input.tomo} {output.filtered_tomo} \
        --process filter.lowpass.gauss:cutoff_abs={params.lowpass_cutoff} \
        --process filter.highpass.gauss:cutoff_pixels={params.highpass_cutoff} \
        --process normalize \
        --process threshold.clampminmax.nsigma:nsigma={params.clamp_nsigma} \
        """

rule slice_tomogram:
    input:
        config = user_config_file,
        tomo = filtered_pattern if config["preprocessing"]["filtering"]["active"] else lambda wildcards: training_meta.loc[wildcards.prefix, "data"],
        labels = lambda wildcards: training_meta.loc[wildcards.prefix, "labels"] # This way the names can differ between tomo and labels
    output:
        sliced_tomo = slices_pattern
    params:
        flip_y = lambda wildcards: training_meta.loc[wildcards.prefix, "flip_y"] * "--flip_y"
    shell:
        f"""
        python3 {srcdir}/scripts/create_training_data.py \
        --features {{input.tomo}} \
        --labels {{input.labels}} \
        --output {{output.sliced_tomo}} \
        --config {{input.config}} \
        {{params.flip_y}} \
        """

rule train_evaluation_model:
    input:
        config = user_config_file,
        training_data = all_training_slices
    output:
        eval_done_file = eval_done_file
    shell:
        f"""
        python3 {srcdir}/scripts/train_eval_model.py \
        --config {{input.config}} \
        --datasets {{input.training_data}} \
        && touch {{output.eval_done_file}} \
        """

rule train_production_model:
    input:
        config = user_config_file,
        training_data = all_training_slices
    output:
        model = config["training"]["production"]["model_output"] if config["training"]["production"]["active"] else []
    shell:
        f"""
        python3 {srcdir}/scripts/train_prod_model.py \
        --config {{input.config}} \
        --datasets {{input.training_data}} \
        """

rule predict_organelles:
    input:
        config = user_config_file,
        tomo = filtered_pattern if config["preprocessing"]["filtering"]["active"] else lambda wildcards: prediction_meta.loc[wildcards.prefix, "data"],
        model = config["training"]["production"]["model_output"] if config["prediction"]["model"] is None else config["prediction"]["model"]
    output:
        prediction = prediction_pattern
    shell:
        f"""
        python3 {srcdir}/scripts/predict_organelles.py \
        --features {{input.tomo}} \
        --output {{output.prediction}} \
        --model {{input.model}} \
        --config {{input.config}} \
        """
