import argparse

parser = argparse.ArgumentParser()

parser.add_argument( 
    "-f",
    "--features",
    required=True,
    nargs='+'
)

args = parser.parse_args()

print(args.features)