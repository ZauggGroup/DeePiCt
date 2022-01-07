import subprocess


def convert_if_none(value: str):
    value = None if value in ['None', "", " "] else value
    return value


def submit_bashCommand(bashCommand: str) -> tuple:
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def slurm_header_writer(my_submission_file: str, group: str, nodes: str,
                        ntasks: str, mem: str, job_time: str, log_file: str,
                        partition: str, error_file: str = None,
                        mail_address: str = None,
                        mail_type: str = None, card: str = None,
                        gres: str = None) -> None:
    dictionary = {
        "#SBATCH -A ": group,
        "#SBATCH --nodes ": nodes,
        "#SBATCH --ntasks ": ntasks,
        "#SBATCH --mem ": mem,
        "#SBATCH --time ": job_time,
        "#SBATCH -o ": log_file,
        "#SBATCH -e ": error_file,
        "#SBATCH --mail-type=": mail_type,
        "#SBATCH --mail-user=": mail_address,
        "#SBATCH -p ": partition,
        "#SBATCH -C ": card,
        "#SBATCH --gres=": gres,
    }
    with open(my_submission_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("#! /bin/bash" + "\n")

        for key in dictionary.keys():
            if dictionary[key] is not None:
                line = key + dictionary[key]
                f.write(line + "\n")
        f.write('\n \n' + content)
    return


if __name__ == "__main__":
    import argparse

    print("Adding slurm header to script")
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", help="file", type=str)
    parser.add_argument("-group", "--group", help="group", type=str)
    parser.add_argument("-nodes", "--nodes", help="nodes ", type=str)
    parser.add_argument("-ntasks", "--ntasks", help="ntasks", type=str)
    parser.add_argument("-mem", "--mem", help="mem", type=str)
    parser.add_argument("-job_time", "--job_time", help="job_time", type=str)
    parser.add_argument("-log_file", "--log_file", help="log_file", type=str)
    parser.add_argument("-error_file", "--error_file", help="error_file",
                        type=str, default=None)
    parser.add_argument("-mail_address", "--mail_address", help="mail_address",
                        type=str, default=None)
    parser.add_argument("-mail_type", "--mail_type", help="mail_type", type=str,
                        default=None)
    parser.add_argument("-partition", "--partition", help="partition", type=str)
    parser.add_argument("-card", "--card", help="card", type=str, default=None)
    parser.add_argument("-gres", "--gres", help="gres", type=str, default=None)

    args = parser.parse_args()

    file = convert_if_none(args.file)
    group = convert_if_none(args.group)
    nodes = convert_if_none(args.nodes)
    ntasks = convert_if_none(args.ntasks)
    mem = convert_if_none(args.mem)
    job_time = convert_if_none(args.job_time)
    log_file = convert_if_none(args.log_file)
    error_file = convert_if_none(args.error_file)
    mail_address = convert_if_none(args.mail_address)
    mail_type = convert_if_none(args.mail_type)
    partition = convert_if_none(args.partition)
    card = convert_if_none(args.card)
    gres = convert_if_none(args.gres)

    if isinstance(gres, str):
        gres = gres.split(',')
        gres_str = ''
        for gre in gres:
            gres_str += gre + ' '
    else:
        gres_str = None

    slurm_header_writer(my_submission_file=file, group=group, nodes=nodes,
                        ntasks=ntasks, mem=mem, job_time=job_time,
                        log_file=log_file, mail_address=mail_address,
                        error_file=error_file, mail_type=mail_type,
                        partition=partition, card=card, gres=gres_str)
