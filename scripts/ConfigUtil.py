import yaml
import pprint

def recursive_get(d, k):
    if len(k) == 1:
        return d[k[0]]
    else:
        return recursive_get(d[k[0]], k[1:])

def csv_list(csv_str):
    return [int(i) for i in csv_str.split(",")]

def assemble_config(defaults, user_config = None, subconfig_paths = None, cli_args = None):
    """
    Assemble script configuration from defaults, user-supplied config file and CLI arguments.
    Override priority: defaults < config file < args
    """
    config = {}

    with open(defaults, 'r') as f:
        defaults = yaml.safe_load(f)
    
    for s in subconfig_paths:
        subconfig = recursive_get(defaults, s)
        config.update(subconfig)
    config["debug"] = defaults["debug"]

    if user_config:
        with open(user_config, 'r') as f:
            user_config = yaml.safe_load(f)
        for s in subconfig_paths:
            subconfig = recursive_get(user_config, s)
            config.update(subconfig)
        config["debug"] = user_config.get("debug", config["debug"])

    if cli_args:
        args_dict = {k:v for k, v in vars(cli_args).items() if v is not None}
        config.update(args_dict)

    if config["debug"]:
        pprint.pprint(config)

    return config
