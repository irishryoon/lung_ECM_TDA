# Main analysis

This directory contains data and code for the main analysis involving 400 ROIs.

## Notebooks & directories
Most notebooks use Julia

* `01_sample_ROIs.ipynb`
	* Code for randomly sampling 400 ROIs of size 4,000 pixels x 4,000 pixels (878 micrometers x 878 micrometers) 
	* The larger images prior to sampling process are not committed to the repository due to file size.
	* The ECM images for sampled 400 ROIs are in `data/4000x4000_combined/subregion_ECM_reduced`
* `02_sample_points_from_ECM.ipynb`
	* Code for sampling point clouds from ECM images. The ECM images are in `data/4000x4000_combined/subregion_ECM`. The original ECM images are too large to be committed to the repository. `data/4000x4000_combined/ROI_ECM_reduced`shows the ECM images, but with reduced file sizes.
	* Point clouds saved in `data/4000x4000_combined/ECM_sampled`
* `03_cells_PH_computation.ipynb`
	* Code for computing PH from cells (cancer cells, leukocytes).
	* PH output saved in `data/4000x4000_combined/cells_PD`
* `04_cells_PH_analysis.ipynb`
	* Analysis (dimensionality reduction) of PH features from cells (cancer cells, leukcoytes).
	* Outputs saved in `analysis/cancer` and `analysis/leukocytes`
* `05_ECM_PH_computation.ipynb`
	* Code for computing PH from ECM point cloud.
	* PH output saved in `data/4000x4000_combined/ECM_PD`
* `06a_ECM_PH_analysis.ipynb`
	* Performs dimensionality reduction of PH features from ECM.
	* Outputs saved in `analysis/ECM`
* `06b_ECM_UMAP_clustering.ipynb`
	* Performs UMAP and clustering on concatenated dim-0 and dim-1 PH features from ECM.
	* Uses the optimal parameters for UMAP.
	* Written in Python.
	* This is the UMAP and clustering reported in the manuscript.
	* Outputs saved in `analysis/ECM/combined_UMAP_clusters` 
* `07_Dowker_PH_computation.ipynb`
	* Due to computation speed, we subsample points when computing Dowker PH. The subsampled points are in the following:
		* `data/4000x4000_combined/Dowker/ECM`
		* `data/4000x4000_combined/Dowker/cancer`
		* `data/4000x4000_combined/Dowker/leukocytes`
	* The Dowker persistence diagrams are saved in
		* `data/4000x4000_combined/Dowker/ECM_cancer`
		* `data/4000x4000_combined/Dowker/cancer_leukocytes`
		* `data/4000x4000_combined/Dowker/ECM_leukocytes`
* `08_Dowker_PH_analysis.ipynb`
	* Performs dimensionality reduction on Dowker PH features. 
	* Outputs saved in `analysis/ECM_cancer`, `analysis/ECM_leukocytes`, `analysis/cancer_leukocytes`
* `09_compute_PH_airways.ipynb`
	* Code for computing PH features on the combined point cloud of ECM, cancer cells, and leukocytes.
	* PH output saved in `data/4000x4000_combined/all_cells_PD`
*  `10_combined_PH_features.ipbny`
	* Code for concatenating all PH and Dowker PH features. 
	* Outputs saved in `analysis/combined/` 