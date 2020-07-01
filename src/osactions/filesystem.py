import os
from os import listdir, makedirs
from os.path import isfile, join, exists

import constants.dirs as dirnames


def get_txt_files(directory_path):
    return [f for f in listdir(directory_path)
            if (isfile(join(directory_path, f))
                and f[0] != '.'
                and str(f).endswith(".txt"))]


def create_dir(path):
    if not exists(path):
        makedirs(path)
    return


def create_dirs(paths):
    for path in paths:
        create_dir(path)
    return


def create_output_folder_tree(root_path):
    create_dir(root_path)
    txt_files_dir_pos = join(root_path, dirnames.PARTICLES_DIR_NAME,
                             dirnames.POS_DIR_NAME,
                             dirnames.TXT_DIR_NAME)
    txt_files_dir_pos_seg = join(root_path, dirnames.PARTICLES_DIR_NAME,
                                 dirnames.POS_SEG_DIR_NAME,
                                 dirnames.TXT_DIR_NAME)
    hdf_files_dir_pos = join(root_path, dirnames.PARTICLES_DIR_NAME,
                             dirnames.POS_DIR_NAME,
                             dirnames.HDF_DIR_NAME)
    hdf_files_dir_pos_seg = join(root_path, dirnames.PARTICLES_DIR_NAME,
                                 dirnames.POS_SEG_DIR_NAME,
                                 dirnames.HDF_DIR_NAME)

    dirs = [txt_files_dir_pos, txt_files_dir_pos_seg,
            hdf_files_dir_pos, hdf_files_dir_pos_seg]

    create_dirs(dirs)
    return dirs


def create_negative_output_folder_tree(root_path):
    create_dir(root_path)
    txt_files_dir = join(root_path, dirnames.PARTICLES_DIR_NAME,
                         dirnames.NEG_DIR_NAME,
                         dirnames.TXT_DIR_NAME)
    hdf_files_dir = join(root_path, dirnames.PARTICLES_DIR_NAME,
                         dirnames.NEG_DIR_NAME,
                         dirnames.HDF_DIR_NAME)

    dirs = [txt_files_dir, hdf_files_dir]

    create_dirs(dirs)
    return dirs


def extract_file_name(path_to_file: str) -> str:
    name = os.path.basename(path_to_file)
    return os.path.splitext(name)[0]
