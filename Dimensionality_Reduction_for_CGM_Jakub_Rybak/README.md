# Dimensionality Reduction for Studying Diffuse Circumgalactic Medium

This repository contains code for the above-mentioned GSOC project. All input data and outputs (as well as the code) can be found here: https://drive.google.com/drive/folders/1qyZcVPA2E1cGRBT2Kt-Lrmj7t7QddfuI?usp=sharing (stored on Google Drive due to size limits).

The project is described in more detail in the following blog post: https://medium.com/@jbrybak/dimensionality-reduction-for-galaxy-evolution-82235391dcd3

# Project description

## Objective
CGM is a gas halo surrounding a galaxy, which contains information on galaxy evolution and history. When a light from a bright source (such as quasar) passes through the gas, part of it gets absorbe, resulting in absorption lines in the spectrum of light intensity, as the figure below shows. It is plausible that there are a few unobserved physical properties of the CGM that determine how the spectrum looks like. This motivates the search for a lower-dimensional latent space. We use a variety of dimensionality-reduction methods to achieve this.

![image](https://user-images.githubusercontent.com/71390120/131004001-9958b083-11c0-4a62-aedf-073a7b629ad1.png)

## How to choose a latent space

Different dimensionality reduction methods result in different latent space. We judge how "good" a given latent space is by how well we can reconstruct the original spectrum. Specifically, we seek to minimise the mean squared reconstruction error.

Below is a comparison of PCA, sparse PCA and Autoencoder in terms of reconstruction error for different latent space sizes. We see that autoencoder achieves low reconstruction error for a small latent space, and therefore is our preferred method.
![image](https://user-images.githubusercontent.com/71390120/131005807-9511753b-6671-470f-95df-f8adbda4c55c.png)

Next, we compare Autoencoder to Variational Autoencoder and as the figure below shows, the two models have almost identical performance. Also, latent space dimension of four seem to be sufficient to capture the input data well.

![image](https://user-images.githubusercontent.com/71390120/131004873-82f0b157-d92f-4e90-ba9a-1421d2fa5805.png)

## Latent space visualisation

Below are scatter plots of the four latent features. As the plot shows, the latent space has meanigful physical interpretation, for example, the spectra with zero impact parameter are clearly separated from those with non-zero impact parameter.

![image](https://user-images.githubusercontent.com/71390120/131009826-c782726e-5980-4695-8f56-9d13ee853642.png)



# Code

The statistical analysis was done on two different datasets: "preliminary" and "updated" dataset. The code for each is contained in the relevant sub-folder of /code folder. The code is non-overlapping, that is, when a certain method was tuned/fitted to both datasets, the code is included only in the "/code/updated_data" folder as we are ultimately interested in analysing this dataset.

The structure of the two folders and content of the individual notebooks and .py files is described below.

In general, all .py files contain functions that are then used inside the notebooks.

## "/code/prelim_data":

**.py files:**
  - data_load.py: contains functions to load data. Given a list of directories, all "hdf5" files are loaded, split into spectral and physical data and concatenated to from a dataset of physical features and a dataset of spectral features.
  - analysis_functions.py: contains functions for plotting, saving model results and statistical analysis (such as backward search in linear model).

**notebooks**

- Generally, notebooks called "model_..." load data, estimate a model and save the output (model object and/or reconstructed samples) to "/code/prelim_data/outputs" folder.
- Notebooks called "analysis..." contain further analysis of the saved models, for example performance comparisons.

- The main "model_" notebook of this section is the "model_pca" one. The remaining methods underperformed (trimmed) PCA. The notebooks are listed in decreasing order of importance of the corresponding method.
- "model_pca.ipynb": Estimates both PCA and trimmed PCA models. It calculates reconstruction loss for these models for a wide range of latent space dimensions, calculates reconstructions using 50 principal components (as this number was chosen to be optimal based on performance analysis) and calculates distance preservation between samples (this was not used much in the end). It also includes Appendix with further analysis of PCA errors, as this method has been extensively studied during this project. 
- "model_pca_tau.ipynb": Estimates PCA using the tau parameter to ensure non-negative reconstructions.
- "model_sparse_pca.ipynb": Estimates sparse PCA model for different penalty coefficients and different latent space sizes. Please note that this is very computationally intensive.
- "model_random_projections.ipynb": Uses random projections method for dimensionality reduction.

- "analysis_reconstruction_loss.ipynb": Compares reconstruction losses for different latent space dimensions across all methods estimated in the notebooks above, plus autoencoder, Conv-1D AE and variational AE models.
- "analysis_pca_ae_errors.ipynb": As AE and (trimmed) PCA seem to be most promising in terms of reconstruction loss, their reconstruction errors are studied more extensively here. Specifically, this notebook looks at scaling of error with volatility of spectrum, potential relations to physical variables.


## "/code/updated_data":

**.py files:**
  - data_load_new.py: Contains functions to load data. Given a list of directories, all "hdf5" files are loaded and concatenated to from physical and spectral datasets dataset. It differs from "data_load" function described above in how the split into physical and spectral data is done (as the input data have different structures).
  - analysis_functions.py: Contains functions for plotting, saving model results and statistical analysis (such as backward search in linear model).
  - model_functions.py: Functions for constructing and tuning AE and VAE models.  

**notebooks**
  - Notebooks are structured slightly differently than in "prelim_data" section: analysis of errors is now within model notebooks. For example, analysis of VAE errors is in the "model_vae.ipynb". The reason for this is that we only have two methods that we analyse separately, rather than many different methods we need to compare.
  - "analysis_updated_data.ipynb": Contains analysis of inpout data, wtithout considering any dimensionality reduction model. Namely, it looks at different types of spectra and illustrates the importance of the impact parameter.
  - Model notebooks have similar structure here. Namely:
  1. Load custom functions: load functions from .py files
  2. Load data
  3. Fit model: Estimate models specified by a grid of hyperparameters and latent space sizes. Checkpoint each model's weights in the "outputs" folder - this allows us to work with the models thereafter without rerunning the fitting procedure.
  4. Load saved model: Load selected model(s) from outputs folder (this involves constructing model object and then loading saved weight from "outputs" folder).
  5. Model selection (**VAE only**): Hyperparameter tuning (namely penalty/KL-divergence weight).
  6. Analysis of latent space: Structure of latent space and relation to physical properties
  7. Further analysis: Error analysis, reconstructed spectra etc.
