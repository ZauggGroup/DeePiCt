from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from file_actions.readers.star import class3d_data_file_reader


def _extract_classes(data_list: list) -> np.array:
    classes = [int(line[13]) for line in data_list]
    return np.array(classes)


def _extract_classes_per_job(job_path: str) -> np.array:
    star_motl_file_name = join(job_path, 'run_it025_data.star')
    data_list = class3d_data_file_reader(star_motl_file_name)
    return _extract_classes(data_list)


def _generate_job_title_string(job_parameters: tuple, param: str) -> str:
    job, K, par = job_parameters
    plot_title = 'J ' + job[-3:] + ', K ' + K + ", " + param + " " + par
    return plot_title


def _generate_class_name_coords_tuple(classes: np.array, k: int) -> tuple:
    class_name = 'class ' + str(k)
    return class_name, np.where(classes == k)


def generate_histograms_for_class3d(bins: int, K_max: int, param: str,
                                    jobs_parameters: dict,
                                    pdf_file_output: str):
    njobs = len(jobs_parameters)
    directory_path = jobs_parameters[0][0][:-6]

    fig, axes = plt.subplots(nrows=njobs, ncols=K_max,
                             figsize=(2.75 * K_max, 3 * njobs))
    fig.suptitle(directory_path, fontsize='x-large')

    subplot_row = 1
    for job_parameters in jobs_parameters:
        job, _, _ = job_parameters
        plot_title = _generate_job_title_string(job_parameters, param)
        classes = _extract_classes_per_job(job)
        class_coords_list = []
        for k in range(K_max):
            class_coords = _generate_class_name_coords_tuple(classes, k + 1)
            class_coords_list += [class_coords]

        for k in range(K_max):
            plt.subplot(njobs, K_max, subplot_row + k)
            class_name, class_coords_values = class_coords_list[k]
            plt.hist(class_coords_values, bins=bins)
            plt.title(plot_title + ', ' + class_name)

        plt.tight_layout()
        subplot_row += K_max

    plt.subplots_adjust(top=0.9)
    plt.show()
    fig.savefig(pdf_file_output)
    return
