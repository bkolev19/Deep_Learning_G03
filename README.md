# Deep Learning Reproduction
This repository aims to reproduce the paper "Data-driven discovery of coordinates and governing equations" by K. Champion et al. For more information, check the blog file in this link: https://bkolev19.github.io/Deep_Learning_G03/.

By Boyan, Ana, Jasper, and Nikolaus

## Libraries
This is the set of packages used for running the repository. Some other versions might crash.
| Package               | Version   | Build                |
|-----------------------|-----------|----------------------|
| numpy                 | 1.21.6    | pypi_0    pypi       |
| pandas                | 1.3.5     | py37h6214cd6_0       |
| matplotlib            | 3.5.3     | py37haa95532_0       |
| scipy                 | 1.7.3     | py37h7a0a035_2       |
| h5py                  | 3.8.0     | pypi_0    pypi       |
| tensorflow-gpu        | 1.15.0    | pypi_0    pypi       |
| keras-preprocessing   | 1.1.2     | pypi_0    pypi       |
| pytorch-cuda          | 11.8      | h24eeafa_5    pytorch|
| torchvision           | 0.11.2+cu113 | pypi_0    pypi    |
| cuda-cudart           | 11.8.89   | 0          nvidia    |
| libcublas             | 11.11.3.6 | 0          nvidia    |
| libcusolver           | 11.4.1.48 | 0          nvidia    |
| ipython               | 7.31.1    | py37haa95532_1       |
| jupyter_core          | 4.11.2    | py37haa95532_0       |
| certifi               | 2022.12.7 | py37haa95532_0       |
| cryptography          | 39.0.1    | py37h21b164f_0       |
| openssl               | 1.1.1w    | h2bbff1b_0           |

## Repository structure
- .gitignore             Describes which files and directories Git should ignore.
- Paper.pdf              The academic paper used as a basis.
- autoencoder.py         Python script for implementing the autoencoder neural network class.
- sindy_utils.py         Python utility script for Sparse Identification of Nonlinear Dynamics (SINDy) methods.
- training.py            Python script for training models, includes cuda methods, feed dictionary, and train_network.
- Lorenz/                Directory with running example.
  - example_lorenz.py    Script with utilities functions for Lorenz data.
  - lorenz_analyse.ipynb Notebook analysing the resulting models.
  - train_lorenz.ipynb   Notebook running the models for training.
  - *Others              Files created after running the training, used for analysis.
