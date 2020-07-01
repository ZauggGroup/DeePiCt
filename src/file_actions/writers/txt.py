def write_model_description(file_path: str,
                            training_data_path: str,
                            label_name: str,
                            split: float,
                            model_name_pkl: str,
                            conf: dict,
                            data_order: list = [],
                            skip_training_set: int or list = None) -> None:
    with open(file_path, 'w') as txt_file:
        txt_file.write("training_data_path = " + training_data_path)
        txt_file.write("\n")
        txt_file.write("label_name=" + label_name)
        txt_file.write("\n")
        txt_file.write("split = " + str(split))
        txt_file.write("\n")
        txt_file.write("model_name = " + model_name_pkl)
        txt_file.write("\n")
        txt_file.write("conf = " + str(conf))
        txt_file.write("\n")
        txt_file.write("data_order = ")
        if isinstance(data_order, int) == 0:
            txt_file.write("Data order not recorded. ")
            txt_file.write("Probably respected order from partition file.")
            txt_file.write("\n")
        else:
            txt_file.write(str(data_order))
            txt_file.write("\n")
        txt_file.write("Skipped data in training list with index =" + str(
            skip_training_set))
    return



