# CryoET Organelle Detection

Modular Snakemake pipeline for Deep Learning-powered detection of organelles in CryoET data using the U-Net architecture by [Ronneberger *et al.*](https://arxiv.org/abs/1505.04597)

## Installation
1) Clone this repo.
2) Install [Miniconda3](https://conda.io/en/latest/miniconda.html). As the pipeline utilizes Snakemake's conda functionalities to install dependencies using Miniconda is a requirement.
3) Install the latest version of [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html) and [Pandas](https://pandas.pydata.org/getting_started.html).
4) (Recommended) install [Keras](https://keras.io/) and [TensorFlow 2.0](https://www.tensorflow.org/) and ensure their functionality. Snakemake will install both packages automatically when running the pipeline, but installing them manually and ensuring that they work properly can save you from a headache when troubleshooting.

## Running the pipeline
Executing the pipeline can be done either locally using [`deploy_local.sh`](deploy_local.sh) or on a Slurm-based cluster using [`deploy_cluster.sh`](deploy_cluster.sh). Running the pipeline also requires a configuration YAML file and one or multiple metadata CSV files (which have to be stated in the config file, see "Configuration").
Apart from that, execution is straightforward:
```bash
$ bash deploy_local.sh <config_file>
```
Deployment scripts may be also altered to adjust the number of jobs run in parallel, do a dryrun etc.

## Input file structure
Features and labels for the pipeline can be either in MRC or REC format, while outputs will be in MRC format.
Data is passed to the pipeline using metadata CSV files containing one row per tomogram.
The CSV file has to contain the following columns:
- **data:** Path to input features. This is the only required column for prediction data.
- **labels:** Path to the respective labels for training data.
- **flip_y:** Whether to flip the sample's labels along the y axis. This is sometimes required when handling annotations from different sources.
- **id:** unique identifier for the tomogram to be used as a prefix for intermediate and output files. This will be used when a working/output directory is specified in the config file. This can be used when input files from different source directories have the same file name. This column is optional.

## Configuration
The pipeline can be easily customized using the configuration file. 
An example configuration with recommended parameters can be found in [`config.yaml`](config.yaml).
Only the *data* section of the config file is required.
All other sections can be omitted, which will result in the defaults (see [`scripts/defaults.yaml`](scripts/defaults.yaml)) being used instead.

### Data
This is the only required section and contains information on where to find and store training data, predictions and intermediate files. An example can be found in [`example_metadata.csv`](example_metadata.csv).
- **training_data:** Metadata CSV file containing location of training features and labels. Only required when training a model.  
- **prediction_data:** Metadata CSV file containing location of features for prediction. Only required when creating predictions.  
- **train_workdir:** Location where intermediate files should be saved. Set to `null` to save at the same location as the source file.
- **output_dir:** Location where predictions should be saved. Set to `null` to save at the same location as the source file.

### Cluster
This section contains parameters only used in cluster execution.
- **logdir:** Where to save cluster execution jobs. By default, this will be the current working directory.

### Preprocessing
This section contains cofiguration for the pipeline multiple preprocessing steps.
- **filtering:** Tomograms can be normalized to have similar contrast by matching their fourier amplitude spectrums.
The algorithm is described [here](https://github.com/mrmattuschka/tomo-spectrum-matcher).
  - **active:** Whether to perform filtering.
  - **target_spectrum:** Target spectrum CSV file to match tomograms to. Create using [`scripts/extract_spectrum.py`](scripts/extract_spectrum.py).
  - **lowpass_cutoff:** Lowpass frequency cutoff in fourier space. Set to 0 to disable.
  - **smoothen_cutoff:** Smoothen cutoff into a sigmoid cuve. Set to 0 to disable.
- **remapping:** This section can be used to remap multiclass annotations into binary labels or drop specific multiclass labels.
  - **active:** Whether to perform remapping.
  - **mapping:** Mapping in `from_label:to_label` format, with one mapping per line. Use `.` as the from-label to remap all non-zero labels that have not received an individual remapping.
- **slicing:** For training, the tomograms are sliced into 2D patches.
  - **patch_size:** Size of 2D patches used for training in `[height, width]` format.
  - **patch_dim:** How many patches to create from a single slice in `[rows, columns]` format. Rows and columns are evenly spaced out to cover the entire slice.
  - **z_cutoff:** Use n/2 slices above and below z center of the tomogram. If `null`, select all labeled slices.
  - **z_stride:** Only use every n-th z slice, as using every slice may just bloat the dataset with redundant information.
  - **crop:** Crop input data before processing into patches.

### **Training**
This section configures training of models for either performance evaluation or prediction.
- **general:** Hyperparameters applicable for both evaluation and production models.
  - **normalize:** Scale the data to be centered on 0 with a standard deviation of 1.
  - **lr:** Learning rate for training.
  - **drop_empty:** Fraction of all-empty slices to drop. This can currently be set to 1 as the Dice loss used in model training is unable to learn from all-empty slices.
  - **batch_size:** Batch size.
  - **n_filters:** Number of filters on first UNet layer.
  - **flip:** Randomly flip tomograms along X/Y axis.
  - **rotate:** Randomly rotate tomograms in 90-degree increments.
- **evaluation:** Parameters for training models for evaluation.
  - **active:** Whether to train evaluation models.
  - **cv_folds:** Cross validation folds (number of tomograms >= cv folds >= 2).
  - **epochs:** Number of training epochs / max. number of epochs if using early stopping.
  - **stopping_patience:** Stop training early if the validation loss does not improve after n epochs, set to 0 to disable.
  - **tensorboard:** Log training metrics to Tensorboard.
  - **run_name:** Run name used in tensorboard, set to `null` to use a timestamp instead.
  - **tf_logdir:** Tensorboard log / output metrics save location.
  - **random_seed:** Set a random seed to make splitting of CV folds reproducible. Set to `null` to disable.
  - **save_models:** Whether to save each CV fold's best model.
  - **model_dir:** Directory to put evaluation per-fold models into.
- **production:** Parameters for training production models (used for creating new predictions).
  - **active:** Whether to train a production model.
  - **epochs:** Number of training epochs. Use the evaluation mode mentioned above to determine a good value for this.
  - **model_output:** Where to save the final production model as a hdf5 file.

### **Prediction** 
Parameters to create predictions from new tomograms using a trained model. The tomogram will be disassembled into 2D patches just like the training data and predictions will be reassembled into 3D stacks.
  - **active:** Whether to create predictions.
  - **model:** Model hdf5 file to use, set to `null` to use the `model_output` stated in the section above.
  - **normalize:** Scale the data to be centered on 0 with a standard deviation of 1.
  - **crop:** Crop patches before reassembly to avoid artifacts at patch edges.
  - **z_cutoff:** Only predict n/2 slices above and below z center.
  - **compensate_crop:** Compensate patch cropping and z cutoff to result in same size as input tomogram.
  - **patch_size:** Must be the same as used for training.
  - **patch_dim:** How many patches to create from a single slice in `[rows, columns]` format. Rows and columns are evenly spaced out to cover the entire slice.

### **Postprocessing** 
Parameters for 3D postprocessing of predictions. 3D predictions assembled from 2D patches may suffer from a large number of false positives, postprocessing the predictions in 3D allows to "iron them out".
- **active:** Whether to perform postprocessing.
- **sigma:** Sigma for 1D gaussian filter along z-axis.
- **threshold:** Threshold to apply to convert from probabilities into binary labels. Set to 0 to disable.
