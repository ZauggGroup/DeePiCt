# 3D UNet adapted for cryo-ET segmentation and particle detection.
Manuscript in preparation.

Authors: Irene de Teresa, Sara K. Goetz, Alexander Mattausch, Christian Zimmerli, Julia Mahamid and Judith Zaugg.

European Molecular Biology Laboratory, Heidelberg, Germany.

## Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [How to run](#How to run)
## 1. Introduction <a name="Introduction"></a>
With improved instrumentation and sample preparation protocols, a large number of high-quality Cryo-ET images are rapidly being generated in laboratories, opening the possibility to conduct high-throughput studies in Cryo-ET. However, due to the crowded nature of the cellular environment together with image acquisition limitations, data mining and image analysis of Cryo-ET tomograms remains one of the most important bottlenecks in the field.
This implementation of a 3D-Unet and subsequent postprocessing steps, has been developed for the task of particle localisation, that enables automating the difficult task of accurately localising macromolecular complexes in Cryo-ET images. 

Figure 1. Segmentation of fatty-acid synthases (FAS), ribosomes and membranes in a S.pombe cryo-tomogram.

![Segmentation of fatty-acid synthases (FAS), ribosomes and membranes in a cryo-tomogram from S.pombe](https://github.com/irenedet/3d-unet/blob/master/images/repo-image.001.png?raw=true)

## 2. Installation<a name="Installation"></a>

### Requirements and conda environment


Package Installation (miniconda and torch)
Install the packages (preferably in a server accessible both from the cluster and locally):

#### Miniconda

```bash
cd foldertodownload
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
during installation, you'll be asked where to install miniconda. Select a folder with large capacity:

```bash
/path/to/folder/with/latge/capacity/miniconda3
```
also, create another folder called .conda in a similar large capacity location:

```bash
mkdir /path/to/folder/with/latge/capacity/.conda
ln -s /path/to/folder/with/latge/capacity/.conda /home/user
```

and add it to the ~/.bashrc file:

```bash
nano ~/.bashrc

# add the line:
export CONDA_PKGS_DIRS=/path/to/large/folder/envs/.conda
```


### Virtual environment
#### Create a conda environment with snakemake
The necessary virtual environment for the installation needs only snakemake and pandas

```bash
conda create -c conda-forge -c bioconda -c conda-forge -n snakemake snakemake==5.13.0
conda activate snakemake
conda install pandas
```



Install pytorch locally:

Note: Install pytorch for GPUs! 
```bash
conda install -c pytorch pytorch torchvision
```



### Clone the 3d-unet repository

```bash
cd /folder/where/the/repository/will/be/cloned
git clone https://gitlab.embl.de/trueba/3d-unet 
```
the repository will be copied to ```/folder/where/the/repository/will/be/cloned/3d-unet```.


Go to the folder where you plan to run the experiments. Create a
configuration file -with the structure given in the example-. Run the pipeline by:

```bash /path/to/3d-unet/deploy_cluster.sh  /path/to/config.yaml```


## How to run<a name="How to run"></a>

### 1. Configuration file structure
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

