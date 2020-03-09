import os.path as op
import os, re, json

################CONFIGURATION################

src_dir   = config["src_dir"]
run_dir   = config["run_dir"]
logdir    = '.'
sendmail  = ''

#############################################

conditions = [ 
    'DM.GLUCOSE25MM',
    'DM.MALTOSE',
    'DM.MELIBIOSE',
    'M9.3HYDROXYCINNAMICACID13P3MM',
    'MOPS.AAFB',
    'MOPS.GLUCOSE',
    'MOPS.XANTHOSINE2P38MM',
    'MOPSC.FUMARICACID',
    'MOPSN.LGLUTAMINE',
    'MOPSP.GLUCOSE6PO4'
]

run_names = [op.splitext(py)[0] for py in os.listdir(run_dir) if op.splitext(py)[1] == '.py']

run_file_pattern = f"{run_dir}/{{run_name}}.py"
done_file_pattern_by_run = f"{run_dir}/{{run_name}}_{{condition}}.done"

all_done_files = expand(done_file_pattern_by_run, run_name=run_names, condition=conditions)


def extract_run_params(file):
    with open(file, "r") as run_file:
        config = re.search("SNAKE(.*?)EKANS", run_file.read(), re.M | re.S).groups(0)[0]
        config_dict = json.loads(config)
        return config_dict

cc = {run_name: extract_run_params(run_file_pattern.format(run_name=run_name)) for run_name in run_names}

rule all:
    input:
        all_done_files

rule train_model:
    input:
        run_file = run_file_pattern,
    output:
        done_file = done_file_pattern_by_run,
    conda:
        "envs/sklearn-cplex.3_6_0.yml"
    params:
        walltime    = lambda w: cc[w.run_name]['t'],
        nodes       = lambda w: cc[w.run_name]['n'],
        cores       = lambda w: cc[w.run_name]['c'],
        memory      = lambda w: cc[w.run_name]['m'],
        data_dir    = config["data_dir"],
        sendmail    = '',
        logdir      = logdir,
    shell:
        f"""
            python -u {{input.run_file}} {config['data_dir']} {{wildcards.condition}} \
            && touch {{output.done_file}}
        """