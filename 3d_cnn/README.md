# 1. Configuration file structure<a name="config"></a>
In the configuration file, the user will specify all the parameters for the network training, prediction, 
postprocessing or evaluation.
Each part of the pipeline has it's own activate parameter that set to `activate: true` or to `activate: false`
will turn on and off the desired running parts of the pipeline.

#### a. General parameters
```bash
dataset_table: "/path/to/table/metadata.csv"  # Description below     
output_dir: "/path/to/output/dir"             # Output directory (will be created while running)
work_dir: "/path/to/work/dir"                 # Destination directory of intermediate files
model_name: "model_name.pkl"                  # Output model using .pkl extension
```

- The `dataset_table`: path to a metadata table in csv format 
containing both the tomogram information for training and for prediction 
(it's structure is specified in the next section).

It should be a comma separated value file (.csv). Its structure -which can be checked in the example 
yeast_metadata.csv-, is the following:

```bash
tomo_name,tomo,<masking_file>,<class_name>_mask,path_to_motl_clean_<class_name>
tomo_name1,/path/to/tomogram/to/segment.mrc,/path/to/some/masking/file.mrc,/path/to/some/class/mask/for/trainingorevaluation.mrc
tomo_name2, ...
tomo_name3, ...
etc.
``` 
 where the fields marked between `< >` are names that should be specified by the user in the config file.
#### b. Specifying training and prediction sets
- The tomograms that will be used for training/prediction are specified through the 
configuration file parameters:
```bash 
tomos_sets:
  training_list: ["tomo_name1", "tomo_name2", etc]      # Tomograms in dataset_table for training ["tomo1", "tomo2", ...]
  prediction_list: ["tomo_name3", "tomo_name_4", etc]   # Tomograms in dataset_table for prediction ["tomo1", "tomo2", ...]
```
 
#### c. Training
```bash
training:
  active: false                               # set to true/false to turn on/off training
  semantic_classes: ['class1', 'class2', etc] # Names of k semantic classes to be learnt. E.g. ['ribo', 'memb']
  processing_tomo: "tomo"                     # Column name in dataset_table corresponding to raw tomo used for training
  box_shape: 64                               # Box-side length of the partition for training
  min_label_fraction: 0.002                   # Minimum label ratio (between 0 and 1) in each box for training, to ensure presence of classes of interest
  overlap: 12                                 # Thickness of overlap for training partition

  # Unet architecture parameters (Only needed for training)
  unet_hyperparameters:
    depth: 1                             # unet depth (=number of maxpooling layers)
    initial_features: 1                  # number of initial convolutions
    n_epochs: 1                          # training epochs
    split: 0.8                           # proportion of training (vs. validation) set, always between 0 and 1
    BatchNorm: False                     # boolean value
    encoder_dropout: 0                   # dropout for encoder path
    decoder_dropout: 0                   # dropout for decoder path
    batch_size: 4                        # batch size for training
```
#### d. Prediction

```bash
prediction:
  active: true
  processing_tomo: "tomo"                # Column name in dataset table corresp. to tomogram that will be segmented
  semantic_class: 'class2'               # The semantic class to be predicted
```
#### e. Postprocessing

For the moment this is only active when prediction is active
```bash 
# Thresholding clustering and motl generation
postprocessing_clustering:
  active: true
  threshold: 0.5                         # Threshold for the probability score to make the predicted segmentation
  min_cluster_size: 100                  # Minimum number of voxels per cluster
  max_cluster_size: 35000                # Maximum number of voxels per cluster
  calculate_motl: False                  # Get the motl of centroids for each cluster
  ignore_border_thickness: 10            # ignore border for motl generation if calculate_motl is True
  filtering_mask: 'lamella_file'         # column name in metadata table for masking segmentation, e.g. lamella_file
```

The column name `<masking_file>` in the dataset table should be present, but
 it can or cannot be filled. It corresponds to the path to 
to a binary map, where the voxels whose value == 1 will be the only region where the segmentation will be performed. For example, if 
`<masking_file>=lamella_file` then only the voxels in the lamella mask will be segmented. In
the config.yaml file the user must set it as 
```
postprocessing_clustering:
     ...
     filtering_mask: 'lamella_file'
```

#### f. Evaluation
There are two possibilities for evaluation, either to compare discrete 
motive list (motl) points for particle picking, or to compare the segmentation
at the voxel level using the Sorenssen Dice-Coefficient.


##### f.1. Particle picking (discrete precision-recall analysis)
To run the particle picking precision-recall analysis, all tomograms in
the prediction_list should have an associated "clean motl" to which compare
the results. It has to be specified under the column path_to_clean_motl_<class>
where <class> is the one indicated in prediction:
```bash
prediction:
  ...
  semantic_class: <class>
```
(see point d).


- The parameters associated to this are:

```bash
evaluation:
  particle_picking:
    active: false
    pr_tolerance_radius: 10              # radius in voxels for considering two coordinate corresp. to same particle
    statistics_file: "pr_statistics.csv" # statistics file where area under pr curve will be stored
```
##### f.2. Segmentation evaluation (dice coefficient)
For evaluation of the segmentation at a voxel level, we calculate the Dice-Coefficient. For that, is needed
that all the tomograms in `prediction_list` have an associated clean mask,
whose path should be specified in the column <class>_mask of the metadata 
`dataset_table`, and <class> is the one indicated in prediction:
```bash
prediction:
  ...
  semantic_class: <class>
```

- The parameters for segmentation evaluation are:
```bash
evaluation:
...
  segmentation_evaluation:
    active: true
    statistics_file: "dice_eval.csv"     # statistics file where the dice coefficient will be stored
```

 #### Note
 
 During training, 3D models are saved alog with the parameters set in the configuration file. In this way, it is possible to track training conditions for model comparison and to reload the model for inference.
