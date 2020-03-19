import argparse
import numpy as np
import yaml
import mrcfile

def main():
    parser = get_cli()
    args = parser.parse_args()

    input_mrc = mrcfile.open(args.input, permissive=True)
    input_tomo = input_mrc.data
    
    mapping = yaml.safe_load(open(args.mapping))

    output_tomo = np.zeros_like(input_tomo)

    c = [i for i in mapping.keys() if not isinstance(i, str)]

    for old_val, new_val in mapping.items():
        if old_val == ".":
            output_tomo[~(np.isin(input_tomo, c) | (input_tomo == 0))] = new_val
        else:
            output_tomo[input_tomo == old_val] = new_val

    out_mrc = mrcfile.new(args.output, overwrite=True, data=output_tomo)
    out_mrc.set_extended_header(input_mrc.extended_header)

def read_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        return f.data

def get_cli():
    # TODO: CLI documentation
    parser = argparse.ArgumentParser(
        description="Relabel 3D tomogram segmentations."
    )

    parser.add_argument( 
        "-i",
        "--input",
        required=True
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True
    )

    parser.add_argument( 
        "-m",
        "--mapping",
        required=True
    )
    
    return parser


if __name__ == "__main__":
    main()