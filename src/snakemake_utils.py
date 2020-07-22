from random import shuffle
import numpy as np


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