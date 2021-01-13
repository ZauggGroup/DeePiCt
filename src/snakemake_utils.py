import os
from random import shuffle
import numpy as np
import pandas as pd


def generate_cv_folds(tomo_training_list: list, cv_folds: int):
    shuffle(tomo_training_list)
    cv_sets = np.array_split(tomo_training_list, cv_folds)
    cv_training_dict = {}
    cv_evaluation_dict = {}
    for cv_idx, cv_ids in enumerate(cv_sets):
        cum_cv_set = []
        cv_evaluation_dict[str(cv_idx)] = list(cv_ids)
        for cv_idx_, cv_ids_ in enumerate(cv_sets):
            if cv_idx_ != cv_idx:
                cum_cv_set += list(cv_ids_)
        cv_training_dict[str(cv_idx)] = cum_cv_set
    return cv_training_dict, cv_evaluation_dict


def generate_cv_data(cv_data_path: str, tomo_training_list: list, cv_folds: int):
    cv_training_dict, cv_evaluation_dict = generate_cv_folds(tomo_training_list=tomo_training_list, cv_folds=cv_folds)
    cv_data = pd.DataFrame({"fold": list(cv_training_dict.keys()),
                            "cv_training_list": [None for _ in range(cv_folds)],
                            "cv_validation_list": [None for _ in range(cv_folds)]})
    cv_data.set_index("fold", inplace=True)
    for fold in cv_training_dict.keys():
        print("fold:", fold)
        cv_training_tomos = cv_training_dict[fold]
        cv_evaluation_tomos = cv_evaluation_dict[fold]
        cv_data["cv_training_list"][fold] = list2str(my_list=cv_training_tomos)
        cv_data["cv_validation_list"][fold] = list2str(my_list=cv_evaluation_tomos)

    os.makedirs(os.path.dirname(cv_data_path), exist_ok=True)
    cv_data.to_csv(cv_data_path)
    return


def list2str(my_list: list) -> str:
    my_str = ""
    for item in my_list:
        my_str += item + " "
    return my_str
