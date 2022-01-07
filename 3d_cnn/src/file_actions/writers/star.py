import csv
import re
from os.path import join

from file_actions.readers.star import class3d_data_file_reader
from osactions.filesystem import create_dir
from relion_toolbox.utils import get_particles_list, \
    get_list_of_indices_and_classes


def _write_star_file_header(csv_writer: csv.writer):
    csv_writer.writerow(["data_"])
    csv_writer.writerow(["loop_"])
    csv_writer.writerow(["_rlnMicrographName", "#1"])
    csv_writer.writerow(["_rlnCoordinateX", "#2"])
    csv_writer.writerow(["_rlnCoordinateY", "#3"])
    csv_writer.writerow(["_rlnCoordinateZ", "#4"])
    csv_writer.writerow(["_rlnImageName", "#5"])
    csv_writer.writerow(["_rlnCtfImage", "#6"])
    csv_writer.writerow(["_rlnGroupNumber", "#7"])
    csv_writer.writerow(["_rlnAngleRot", "#8"])
    csv_writer.writerow(["_rlnAngleTilt", "#9"])
    csv_writer.writerow(["_rlnAnglePsi", "#10"])
    csv_writer.writerow(["_rlnOriginX", "#11"])
    csv_writer.writerow(["_rlnOriginY", "#12"])
    csv_writer.writerow(["_rlnOriginZ", "#13"])
    return


def write_star_from_particles_index_list(new_star_path: str, old_star_path: str,
                                         particles_indices_and_classes: list,
                                         selected_classes: list):
    with open(new_star_path, 'w') as csvfile:
        star_writer = csv.writer(csvfile, delimiter=' ',
                                 quoting=csv.QUOTE_NONE, escapechar=' ')
        _write_star_file_header(star_writer)
        original_particles_list = get_particles_list(old_star_path)
        particle_regex = r"par_(\d+).mrc"
        for line in original_particles_list:
            particle_index = int(re.findall(particle_regex, line[4])[0])
            for particle in particles_indices_and_classes:
                if particle[0] == particle_index and particle[1] \
                        in selected_classes:
                    star_writer.writerow(line)
    print("The new star file was writen in " + new_star_path)
    return


def _get_new_star_path(star_data_path: str, selected_classes: list) -> str:
    new_star_path = join(star_data_path[:-19], "filtered_classes")
    create_dir(new_star_path)
    new_star_path = join(new_star_path, "classes_" + str(selected_classes)
                         + ".star")
    return new_star_path


def generate_new_particles_star_files(job_data: tuple,
                                      old_particles_star_file: str):
    star_data_path, _, _, selected_classes = job_data
    new_particles_star_file = _get_new_star_path(star_data_path,
                                                 selected_classes)
    data_list = class3d_data_file_reader(star_data_path)
    particles_indices_and_classes = get_list_of_indices_and_classes(data_list)
    write_star_from_particles_index_list(new_particles_star_file,
                                         old_particles_star_file,
                                         particles_indices_and_classes,
                                         selected_classes)
    return
