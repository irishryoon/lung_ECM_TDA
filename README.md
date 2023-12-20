# Topological analysis of lung adenocarcinoma

This repository contains code for paper [INSERT PAPER]. If you use these methods, please cite as the following.

[INCLUDE BIBTEX]

* Due to the large size of data, not all data files are included in this repository. Please contact Iris Yoon (hyoon@wesleyan.edu) for copies of the data. 

## Install
Download <a href="https://julialang.org/downloads/">Julia</a>

## Code
* `src/ECM_TDA.jl` contains code for computing topological (PH and Dowker PH features)
* <b>Activating project environment</b>: In the root directory, use the following code to activate the project environment.

```
import Pkg
Pkg.activate(".")
```
* Using `src/ECM_TDA.jl`: Use the following code.

```
include("src/ECM_TDA.jl")
using .ECM_TDA
```
* The following notebooks in `main_analysis` illustrate how one can compute topological features:
	* `03_cells_PH_computation.ipynb`
	* `05_ECM_PH_computation.ipynb`
	* `07_Dowker_PH_computation.ipynb`


## Directories
* `main_analysis`: The main analysis 400 ROIs
* `image_resolution_sensitivity`: Analysis of the robustness of PH features to ECM image resolution. 
* `wholeslide`: Analysis of whole-slide images
* `transcriptomics`: Analysis of topological features and transcriptional data
