# 3d-cnn

## 1. Description.

3D UNet adapted for cryo-ET segmentation and particle detection.

## 2. Installation

### Requirements and conda environment


Package Installation (miniconda and torch)
Install the packages (preferably in a server accessible both from the cluster and locally):

#### Miniconda

```bash
cd foldertodownload
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### pytorch and torchvision for gpus:
```bash
conda install -c pytorch pytorch torchvision
```

Create a directory in a large capacity folder where conda will download installed packages:

```bash
# If this is the first time you install a venv,
# make sure to locate it in a large capacity folder

mkdir /path/to/large/folder/envs/.conda
ln -s  /path/to/large/folder/envs/.conda /home/username

```

And add it to the ~/.bashrc file:

```bash
nano ~/.bashrc

# add the line:
export CONDA_PKGS_DIRS=/path/to/large/folder/envs/.conda
```


### Virtual environment

#### Create the conda virtual environment
Now create the virtual environment for this project with all requirements:

```bash
conda env create --file environment.yaml
```

to activate this environment, use:

```
conda activate 3d-cnn
```

to deactivate an active environment, use:

 ``` conda deactivate ```
