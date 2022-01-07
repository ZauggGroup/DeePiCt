from os.path import join


def create_particle_file_name(folder_path: str, img_number: int,
                              coord_indx: int, ext: str) -> str:
    file_name = str(img_number) + 'particle' + str(coord_indx) + '.' + ext
    return join(folder_path, file_name)
