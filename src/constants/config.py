import datetime
import os

import numpy as np
import yaml

from constants.statistics import ModelDescriptor

CV_DATA_FILE = "cv_data.csv"


class Config:
    def __init__(self, user_config_file: str):
        with open(user_config_file, 'r') as user_config:
            config = yaml.safe_load(user_config)
        self.dataset_table = config["dataset_table"]
        self.output_dir = config["output_dir"]
        self.work_dir = config["work_dir"]
        self.model_path = config["model_path"]
        self.model_name = os.path.basename(self.model_path)[:-4]
        # Tomogram lists
        self.training_tomos = config["tomos_sets"]["training_list"]
        self.prediction_tomos = config["tomos_sets"]["prediction_list"]
        # Training
        self.overlap = config["training"]["overlap"]
        self.partition_name = "train_partition"
        self.semantic_classes = config["training"]["semantic_classes"]
        self.processing_tomo = config["training"]["processing_tomo"]
        self.box_size = config["training"]["box_size"]
        self.min_label_fraction = config["training"]["min_label_fraction"]
        self.max_label_fraction = 1
        self.batch_size = config["training"]["batch_size"]
        self.force_retrain = config["training"]["force_retrain"]

        self.da_rounds = config["training"]["data_augmentation"]["rounds"]
        if self.da_rounds > 0 :
            self.da_rot_angle = config["training"]["data_augmentation"]["rot_angle"]
            self.da_elastic_alpha = config["training"]["data_augmentation"]["elastic_alpha"]
            self.da_sigma_gauss = config["training"]["data_augmentation"]["sigma_gauss"]
            self.da_salt_pepper_p = config["training"]["data_augmentation"]["salt_pepper_p"]
            self.da_salt_pepper_ampl = config["training"]["data_augmentation"]["salt_pepper_ampl"]
        else:
            self.da_rot_angle = 0
            self.da_elastic_alpha = 0
            self.da_sigma_gauss = 0
            self.da_salt_pepper_p = 0
            self.da_salt_pepper_ampl = 0

        # self.apply_filter = config["Filter"]["active"]
        # if self.apply_filter:
        #     self.cutoff = config["Filter"]["cutoff"]
        #     self.smoothen = config["Filter"]["smoothen"]

        # unet_hyperparameters:
        self.depth = config["training"]["unet_hyperparameters"]["depth"]
        self.initial_features = config["training"]["unet_hyperparameters"]["initial_features"]
        self.epochs = config["training"]["unet_hyperparameters"]["epochs"]
        self.split = config["training"]["unet_hyperparameters"]["train_split"]
        self.batch_norm = config["training"]["unet_hyperparameters"]["batch_norm"]
        self.encoder_dropout = config["training"]["unet_hyperparameters"]["encoder_dropout"]
        self.decoder_dropout = config["training"]["unet_hyperparameters"]["decoder_dropout"]
        self.model_hyperparameters = config["training"]["unet_hyperparameters"]

        self.pred_class = config["prediction"]["semantic_class"]
        self.pred_class_number = -1
        for class_number, semantic_class in enumerate(self.semantic_classes):
            if semantic_class == self.pred_class:
                self.pred_class_number = class_number
        assert self.pred_class_number >= 0, "Prediction class not among segmentation names for this model!"

        # Thresholding clustering and motl generation
        self.threshold = config["postprocessing_clustering"]["threshold"]
        self.min_cluster_size = config["postprocessing_clustering"]["min_cluster_size"]
        self.max_cluster_size = config["postprocessing_clustering"]["max_cluster_size"]
        if self.max_cluster_size is None:
            self.max_cluster_size = np.inf
        self.clustering_connectivity = config["postprocessing_clustering"]["clustering_connectivity"]
        self.calculate_motl = config["postprocessing_clustering"]["calculate_motl"]
        self.ignore_border_thickness = config["postprocessing_clustering"]["ignore_border_thickness"]
        self.region_mask = config["postprocessing_clustering"]["region_mask"]
        self.contact_mode = config["postprocessing_clustering"]["contact_mode"]
        self.contact_distance = config["postprocessing_clustering"]["contact_distance"]

        # evaluation:
        # a. For precision recall in particle picking
        self.pr_active = config["evaluation"]["particle_picking"]["active"]
        self.pr_tolerance_radius = config["evaluation"]["particle_picking"]["pr_tolerance_radius"]
        self.pr_statistics_file = config["evaluation"]["particle_picking"]["statistics_file"]

        # b. For dice coefficient evaluation at the voxel level
        self.dice_eval_active = config["evaluation"]["segmentation_evaluation"]["active"]
        self.dice_eval_statistics_file = config["evaluation"]["segmentation_evaluation"]["statistics_file"]

        self.old_model = None
        self.retrain = False
        self.total_folds = config["cross_validation"]["folds"] if config["cross_validation"] == 'active' else None


def get_model_name(config: Config, fold: int):
    if fold is not None:
        model_name = config.model_name + "_" + str(fold)
        model_path = config.output_dir + "/models/" + model_name + ".pth"
    else:
        model_path = config.model_path
        model_name = config.model_name
    return model_path, model_name


def record_model(config: Config, training_tomos: list, testing_tomos: list, fold: int = None):
    now = datetime.datetime.now()
    training_date = str(now.day) + "/" + str(now.month) + "/" + str(now.year)
    logging_dir = os.path.join(config.output_dir, "logging")
    model_dir = os.path.join(config.output_dir, "models")
    models_table = os.path.join(model_dir, "models.csv")
    log_path = os.path.join(logging_dir, config.model_name)
    model_descriptor = model_descriptor_from_config(config=config, training_date=training_date, fold=fold,
                                                    testing_tomos=testing_tomos, training_tomos=training_tomos,
                                                    log_path=log_path)

    model_descriptor.add_descriptor_to_table(table_path=models_table, model_descriptor=model_descriptor,
                                             force_retrain=config.force_retrain)
    return model_descriptor


def model_descriptor_from_config(config: Config, training_date=None, fold=None, testing_tomos=None,
                                 training_tomos=None, log_path=None):
    model_path, model_name = get_model_name(config, fold)
    model_descriptor = ModelDescriptor(batch_norm=config.batch_norm, box_size=config.box_size,
                                       training_date=training_date, decoder_dropout=config.decoder_dropout,
                                       encoder_dropout=config.encoder_dropout, depth=config.depth,
                                       initial_features=config.initial_features, log_path=log_path,
                                       model_name=model_name, model_path=config.model_path, epochs=config.epochs,
                                       old_model=config.old_model, output_classes=len(config.semantic_classes),
                                       overlap=config.overlap, partition_name=config.partition_name,
                                       processing_tomo=config.processing_tomo, retrain=config.retrain,
                                       semantic_classes=config.semantic_classes, train_split=config.split,
                                       training_set=training_tomos, testing_set=testing_tomos,
                                       total_folds=config.total_folds, fold=fold, da_rounds=config.da_rounds,
                                       da_rot_angle=config.da_rot_angle, da_elastic_alpha=config.da_elastic_alpha,
                                       da_sigma_gauss=config.da_sigma_gauss, da_salt_pepper_p=config.da_salt_pepper_p,
                                       da_salt_pepper_ampl=config.da_salt_pepper_ampl)
    return model_descriptor
