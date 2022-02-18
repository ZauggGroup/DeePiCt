import mrcfile


def read_mrc(path_to_mrc: str, dtype=None):
    with mrcfile.open(path_to_mrc, permissive=True) as f:
        return f.data
