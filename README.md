# SCSR-core

Official implementation of Stochastic Cortical Self-Reconstruction (SCSR). 

## Installation

1. Create environment: `conda env create -n SCSR --file requirements.yml`
2. Activate environment: `conda activate SCSR`



## Data

We used data from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/).

## Usage

The package uses [PyTorch](https://pytorch.org). To train SCSR, call `python DAE_mlp_train.py config_files/training_configs/config_99.yaml`. 


## Citation

```bibtex
@article{wachinger2024stochastic,
  title={Stochastic Cortical Self-Reconstruction},
  author={Wachinger, Christian and Hedderich, Dennis and Bongratz, Fabian},
  journal={arXiv preprint arXiv:2403.06837},
  year={2024}
}
```
