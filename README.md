# SCSR-core

Official implementation of Stochastic Cortical Self-Reconstruction (SCSR). 

![SCSR logo](images/SCSR_logo.png)


## Installation

1. Check out repository
2. Create environment: `conda env create --file environment.yml`
3. Activate environment: `conda activate SCSR`
4. Download model https://drive.google.com/file/d/1qmD5m3wR1F_sqVmBTyZgWWxi_VCpEbKz/view?usp=sharing and copy to directory `checkpoints`



## Data

We used data from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/).

## Usage

- The package uses [PyTorch](https://pytorch.org)
- As input data, an input table with columns ['DX', 'AGE', 'PTGENDER', per-vertex values] is expected as a .feather file
- To train SCSR, set the path to the input table `table_path` in the training file and call `python SCSR_train.py config_files/training_configs/config.yaml`. 
- For testing, again set the path to the input table `table_path` in the testing file and call `python SCSR_test.py`


## Citation

```bibtex
@article{wachinger2024stochastic,
  title={Stochastic Cortical Self-Reconstruction},
  author={Wachinger, Christian and Hedderich, Dennis and Bongratz, Fabian},
  journal={arXiv preprint arXiv:2403.06837},
  year={2024}
}
```
