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
conda create -c bioconda -n snakemake snakemake
conda activate snakemake
```

Install pytorch locally:

```bash
conda install -c pytorch pytorch torchvision
```


#### Running the pipeline
##### Clone the 3d-unet repository

```bash
cd /folder/where/the/repository/will/be/cloned
git clone https://gitlab.embl.de/trueba/3d-unet 
```
the repository will be copied to ```/folder/where/the/repository/will/be/cloned/3d-unet```.


Go to the folder where you plan to run the experiments. Create a
configuration file -with the structure given in the example-. Run the pipeline by:

```bash /path/to/3d-unet /path/to/config.yaml```
