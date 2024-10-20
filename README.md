# SCSR-core

Official implementation of Stochastic Cortical Self-Reconstruction (SCSR). 

## Installation

1. Create environment: `conda env create -n SCSR --file requirements.yml`
2. Activate environment: `conda activate SCSR`



## Data

We used data from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/).

## Usage

The package uses [PyTorch](https://pytorch.org). To train and test PASTA, execute the `train_mri2pet.py` script. 
The configuration file of the command arguments is stored in `src/config/pasta_mri2pet.yaml`.


## Citation

```bibtex
@InProceedings{Li2024pasta,
    author="Li, Yitong
    and Yakushev, Igor
    and Hedderich, Dennis M.
    and Wachinger, Christian",
    title="PASTA: Pathology-Aware MRI to PET Cross-Modal Translation with Diffusion Models",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="529--540",
    isbn="978-3-031-72104-5"
}
```
