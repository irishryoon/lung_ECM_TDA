# Topological analysis of lung adenocarcinoma

This repository contains code for paper [INSERT PAPER]. If you use these methods, please cite as the following.

[INCLUDE BIBTEX]

## Code
* `src/ECM_TDA.jl` contains code for computing topological (PH and Dowker PH features)
* The notebooks illustrate the following process on 400 ROIs:
	* preprocessing data
	* computing topological features (PH features and Dowker PH features) 
	* analysis of topological features	

## Directories
* `data`: PSRH images and cell (cancer cells and leukcoytes) locations of 400 ROIs
* `analysis`: analysis of topological features from 400 ROIs
* `image_resolution_sensitivity`: Analysis of the robustness of PH features to ECM image resolution. 