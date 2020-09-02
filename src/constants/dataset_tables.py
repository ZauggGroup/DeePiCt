class DatasetTableHeader(object):
    def __init__(self, semantic_classes: list = None,
                 clean_motl: list = None, filtering_mask: str = "lamella_file",
                 partition_name='train_partition',
                 processing_tomo='eman2_filtered_tomo',
                 image_acquisition_parameter='vpp'):

        self.processing_tomo = processing_tomo
        self.original_tomo = 'original_tomo'
        self.tomo_name = 'tomo_name'
        self.semantic_classes = semantic_classes
        self.sample_type = 'sample_type'
        self.binning = 'binning'
        self.image_acquisition_parameter = image_acquisition_parameter
        self.filtering_mask = filtering_mask
        self.species = 'species'
        self.partition_name = partition_name
        self.test_partition = 'test_partition'
        self.x_dim = 'x_dim'
        self.y_dim = 'y_dim'
        self.z_dim = 'z_dim'
        self.x_shift = 'x_shift'
        self.y_shift = 'y_shift'
        self.z_shift = 'z_shift'
        if semantic_classes is not None:
            self.masks_names = [
                name + "_mask" for name in self.semantic_classes
            ]
        if clean_motl is None:
            if semantic_classes is not None:
                self.clean_motls = ["path_to_motl_clean_" + semantic_class for
                                    semantic_class in semantic_classes]
        else:
            self.clean_motls = clean_motl
        return


class ModelsTableHeader(object):

    def __init__(self):
        self.fold = "fold"
        self.total_folds = "total_folds"
        self.testing_set = "testing_set"
        self.batch_norm = "batch_norm"
        self.training_date = "training_date"
        self.depth = "depth"
        self.initial_features = 'initial_features'
        self.log_path = 'log_path'
        self.model_name = "model_name"
        self.model_path = "model_path"
        self.epochs = "epochs"
        self.old_model = "old_model"
        self.output_classes = "output_classes"
        self.retrain = "retrain"
        self.segmentation_names = "semantic_classes"
        self.train_split = "train_split"
        self.training_set = "training_set"
        self.encoder_dropout = "encoder_dropout"
        self.decoder_dropout = "decoder_dropout"
        self.label_name = "label_name"
        self.box_size = "box_size"
        self.partition_name = "partition_name"
        self.processing_tomo = "processing_tomo"
        self.overlap = "overlap"
        self.dtype_dict = {
            self.fold: str,
            self.testing_set: str,
            self.batch_norm: bool,
            self.training_date: str,
            self.depth: int,
            self.initial_features: int,
            self.log_path: str,
            self.model_name: str,
            self.model_path: str,
            self.epochs: int,
            self.old_model: str,
            self.output_classes: int,
            self.retrain: bool,
            self.train_split: float,
            self.encoder_dropout: float,
            self.decoder_dropout: float,
            self.label_name: str
        }
        return

