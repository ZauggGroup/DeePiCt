def class3d_data_file_reader(motl_file_name: str) -> list:
    with open(motl_file_name, "r") as f:
        row_list = [l for l in (line.strip() for line in f) if l]
        data_list = [l.split() for l in row_list if len(l.split()) >= 18]
    return data_list
