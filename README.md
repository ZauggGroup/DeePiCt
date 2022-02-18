# Convolutional networks for supervised mining of intracellular molecular landscapes.

Manuscript in preparation.

Authors: Irene de Teresa*, Sara K. Goetz*, Alexander Mattausch, Frosina Stojanovska, Christian Zimmerli, Mauricio Toro-Nahuelpan, Dorothy W.C. Cheng, Fergus Tollervey, Constantin Pape, Martin Beck, Anna Kreshuk, Julia Mahamid and Judith Zaugg.

## Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [How to run](#How_to_run)
## 1. Introduction <a name="Introduction"></a>
With improved instrumentation and sample preparation protocols, a large number of high-quality 
cryo-ET images are rapidly being generated in laboratories, opening the possibility to conduct 
high-throughput studies in cryo-ET. However, due to the crowded nature of the cellular environment 
together with image acquisition limitations, data mining and image analysis of cryo-ET tomograms 
remains one of the most important bottlenecks in the field.
We present DeePiCt (Deep Picker in Context), a deep-learning based pipeline to achieve structure 
segmentation and particle localization in cryo-electron tomography. DeePiCt combines two dedicated 
convolutional networks: a 2D CNN for segmentation of cellular compartments (e.g. organelles or cytosol),
and a 3D CNN for particle localization and structure segmentation. 

Figure 1. Segmentation of fatty-acid synthases (FAS), 
ribosomes and membranes in a *S. pombe* cryo-tomogram.

![Segmentation of fatty-acid synthases (FAS), ribosomes and membranes in a cryo-tomogram from S.pombe](./images/repo-image.001.png?raw=true)

## 2. Installation<a name="Installation"></a>

Both 2D and 3D CNN pipelines require a conda installation, and are run via the Snakemake workflow management system  
(https://snakemake.readthedocs.io/en/stable/).

### Requirements and conda environment

Package Installation (miniconda, Pytorch and Keras).

#### Miniconda

Download the latest miniconda3 release, according to your OS and processor (modify the Miniconda3-latest-Linux-x86_64.sh
file according to the instructions at https://docs.conda.io/en/latest/miniconda.html):

```bash
cd foldertodownload
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

during installation, you'll be asked where to install miniconda. Select a folder with large capacity:

```bash
/path/to/folder/with/large/capacity/miniconda3
```

### Virtual environment

#### Create a basic conda environment with Snakemake and pandas
The necessary virtual environment for the installation needs only snakemake and pandas

```bash
conda install -n base -c conda-forge mamba
conda activate base
mamba create -c conda-forge -c bioconda -n snakemake snakemake==5.13.0
conda activate snakemake
conda install pandas
```


Install Pytorch:

```bash
conda install -c pytorch pytorch torchvision
```

Install Keras:

```
conda install -c anaconda keras-gpu=2.3.1
```

### Clone this repository

```bash
cd /folder/where/the/repository/will/be/cloned
git clone https://github.com/mrmattuschka/DeePiCt
```

## How to run<a name="How_to_run"></a>

Go to the folder where you plan to run the experiments. Create a
configuration file -with the structure given in the examples (see each of the 
2d_cnn/config.yaml or 3d_cnn/config.yaml files)-. Run the pipeline by:

- 2D CNN pipeline:
```bash /path/to/2d_cnn/deploy_cluster.sh  /path/to/config.yaml```

- 3D CNN pipeline:
```bash /path/to/3d_cnn/deploy_cluster.sh  /path/to/config.yaml```

(notice that, to run locally the deploy_cluster.sh script can be exchanged by deploy_local.sh).

### 1. Configuration file structure
See each of the 2d_cnn/README.md and 2d_cnn/README.md files.