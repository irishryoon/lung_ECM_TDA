module ECM_TDA
include("dowker_persistence.jl") # SWAP WITH DOWKER CODE
using .Dowker
using Eirene
using Plots
using Combinatorics
using Distributions
using DataFrames
using PersistenceDiagrams
using JLD2
using Groups
using FileIO
using Distances
using Images
using CSV
using PyCall
using Ripserer
using TiffImages
using NPZ
using Plots
using Measures
using MultivariateStats
using LinearAlgebra
using Random
using StatsBase
using DelimitedFiles
using UMAP
using KernelDensity
using StatsPlots

export  
        load_ECM_image,
        load_cells,
        n_ECM_samples,
        sample_ECM_points,
        plot_PD,
        plot_cyclerep,
        sample_regions,
        sample_subregion_centers,
        plot_subregions,
        plot_subregion_boundaries,
        get_img_sub,
        run_PH,
        run_PH_cell_type,
        run_PH_whole_image,
        RipsererPD_to_array,
        get_csv_path,
        array_to_ripsererPD,
        get_subimage_dict,
        plot_ROI_ECM_cells,
        compute_PI,
        compute_PI_dissimilarity,
        plot_subregion_boxes,
        get_subregion_boundaries,
        get_PD0_max,
        get_PD1_max,
        plot_profile_dim0,
        plot_profile_dim1,
        plot_Dowker_profile, 
        PI0_to_PCA,
        PI1_to_PCA,
        PI_to_PCA,
        plot_scores,
        get_large_coordinate_examples,
        compute_Dowker_cells,
        compute_Dowker,
        find_idx_from_range,
        sample_uniform,
        sample_kde,
        index_closest_to_x,
        LTX_Da_idx_tuple_to_filename,
        cbar_tickvals_to_loc,
        plot_PI,
        center_PI,
        get_1simplices,
        get_2simplices,
        plot_Dowker_complex,
        plot_dim_red,
        create_features_array_dim0,
        create_features_array_dim1,
        get_ranked_persistence,
        get_coordinate_min_max_examples,
        plot_low_high_PC_cancer_PSRH,
        plot_low_high_PC_cancer_leukocytes_PSRH,
        get_small_large_coordinate_examples,
        plot_Dowker_profile_cells,
        compute_PI2,
        combine_PI0_PI1_dicts_Dowker,
        centered_features_to_PCA,
        PI_to_PCA2,
        plot_PI2_old,
        plot_PI2,
        plot_PSRH,
        plot_dim_red2,
        load_PI_dict

        
        

#-----------------------------------------------------------------------------------
# functions for loading and preprocessing data 
#-----------------------------------------------------------------------------------
function load_cells(LTX, Da; green_purple = :nothing)
    if green_purple == :nothing
        return CSV.read("data/ROI/region_csv_scale_corrected/LTX" * LTX * "PSR_20X.czi/Da" * Da *".csv")
    elseif green_purple == "green"
        return CSV.read("data/ES_201222/stitch_region_40x_green/region_csv/LTX" * LTX * "PSR_20X.czi/Da" * Da *".csv")
    elseif green_purple == "purple"
        return CSV.read("data/ES_201222/stitch_region_40x_purple/region_csv/LTX" * LTX * "PSR_20X.czi/Da" * Da *".csv")
    end   
end

function load_ECM_image(LTX, Da; green_purple = :nothing)
    
    # specify ECM directory
    if green_purple == :nothing
        img_tif = "data/ROI/deconvolved/" *  "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.tif"
        img_jpg = "data/ROI/deconvolved/" *  "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.jpg"
    elseif green_purple == "green"
        img_tif = "data/ROI_ES_201222/deconvolved/stitch_region_40x_green/" * 
                "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.tif"
        img_jpg = "data/ROI_ES_201222/deconvolved/stitch_region_40x_green/" * 
                "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.jpg"  
    elseif green_purple == "purple"
        img_tif = "data/ROI_ES_201222/deconvolved/stitch_region_40x_purple/" * 
                "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.tif"
        img_jpg = "data/ROI_ES_201222/deconvolved/stitch_region_40x_purple/" * 
                "LTX" * LTX * "PSR_20X.czi/deconvolved/roi_Da" * Da * "_deconv.jpg"  
    end
    
    # try opening .tif
    try
        img = Images.load(img_tif)
    catch
        img = Images.load(img_jpg)
    end
    
    # exception for one ROI
    if (LTX =="124") & (Da == "104")
        imgg = Gray.(img)
        mat = convert(Array{Float64}, imgg)
        return mat

    elseif (LTX =="124") & (Da == "385")
        imgg = Gray.(img)
        mat = convert(Array{Float64}, imgg)
        return mat
    else 

        # invert
        img = 1 .- img
    
        return img
    end
end


#-----------------------------------------------------------------------------------
# functions for sampling sub-regions from ROI
#-----------------------------------------------------------------------------------
"""
    sample_regions(
Sample regions from images. 
"""
function sample_regions(ROI_size_x, ROI_size_y, subregion_size, distance_required; n_samples = 2)
    sampled_locations = []
    trial_num = 0
    while size(sampled_locations)[1] < n_samples
        trial_num += 1

        sampled_x = sample(Int(subregion_size//2): Int(ROI_size_x - subregion_size//2), 1)[1]
        sampled_y = sample(Int(subregion_size//2) : Int(ROI_size_y - subregion_size//2), 1)[1]
        
        
        # check that the newly sampled point is far from existing points
        count = 0
        for i = 1:size(sampled_locations)[1]
            x, y = sampled_locations[i]
            if (sampled_x - x)^2 + (sampled_y - y)^2 > distance_required^2
                count += 1
            end
        end
        if count == size(sampled_locations)[1]
            push!(sampled_locations, [sampled_x, sampled_y])     
        end
        
        # if we reach a certain trial number, re-start
        if trial_num == 100
            sampled_locations = []
            trial_num = 0
        end
    end
    return sampled_locations
end

"""
    sample_subregion_centers(ROI_size, subregion_size, distance_required)
Given a dictionary of ROI_sizes, sample locations of subregion centers.

### Inputs
- `ROI_size`: dictionary. ROI_size[(LTX, Da)] = (width, height) of corresponding ROI
- `subregion_size`: size of subregion
- `distance_required`: distance between subregion centers

### Outputs
- `subregion_centers`: a dictionary where subregion_centers[(LTX, Da)] is a list of centers. 
"""
function sample_subregion_centers(ROI_size, subregion_size, distance_required)
    subregion_centers = Dict()
    for (key, val) in ROI_size
        LTX, Da = key
        img_w, img_h = val
        if (img_w >= subregion_size) & (img_h >= subregion_size)
            if (img_w == subregion_size) | (img_h == subregion_size)
                if img_w == subregion_size
                    sampled  = sample(Int(subregion_size//2): Int(img_h - subregion_size//2), 1)[1]
                    sampled_loc = [[Int(subregion_size//2), sampled]]
                end

                if img_h == subregion_size
                    sampled  = sample(Int(subregion_size//2): Int(img_w - subregion_size//2), 1)[1]
                    sampled_loc = [[sampled, Int(subregion_size//2)]]
                end  
            else
                max_length = maximum([img_w, img_h])
                n_samples = floor.(Int,max_length / subregion_size)

                sampled_loc = sample_regions(img_w, img_h, subregion_size, distance_required; n_samples = n_samples)
            end
            subregion_centers[(LTX, Da)] = sampled_loc
        end
    end
    return subregion_centers
end

function plot_subregion_boxes(LTX, Da, subregion_centers; green_purple = :nothing, subregion_size = 4000, lw= 5, kwargs...)
    # get image
    img = load_ECM_image(LTX, Da; green_purple = green_purple)
    
    p = plot(Gray.(img), frame = :box, ticks = []; kwargs...)
    for (idx, center) in enumerate(subregion_centers)
        center_x, center_y = center
        xleft = center_y - Int(subregion_size // 2)
        xright = center_y + Int(subregion_size // 2)
        yleft = center_x - Int(subregion_size // 2)
        yright = center_x + Int(subregion_size // 2)
        plot!(p, [xleft, xright], [yleft, yleft], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xleft, xleft], [yleft, yright], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xright, xright], [yleft, yright], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xleft, xright], [yright, yright], linewidth = lw, color = :deeppink, label = "") 
        scatter!(p, [center_y], [center_x], text = string(idx), label = "", alpha = 0)
    end
    return p
end

#-----------------------------------------------------------------------------------
# functions for sampling points from ECM
#-----------------------------------------------------------------------------------
"""
    n_ECM_samples()
Determines the number of points to sample from mean ECM pixel value. Creates a piecewise-linear function with three pieces,
with f(low) = min_sample and f(high) = max_sample
The argument `x` is mean ECM pixel value, scaled to be [0,1], where x close to 1 means there is higher ECM content.
"""
function n_ECM_samples(x, low, high, min_sample, max_sample)
    if x <= low
        return min_sample
    elseif x <= high
        slope = (max_sample - min_sample) / (high - low)
        return slope * (x - low) + min_sample
    else
        return max_sample
    end
end

"""
    sample_ECM_points(image; <keyword arguments>)
Sample points from an ECM image. 
1. For each pixel (with pixel value p in [0,1]), sample points according to the binomial distribution with p^2 as probability.  
2. Remove "island points" 
3. Downsample
"""
function sample_ECM_points(image; 
    invert = true,
    vicinity = 50,
    n_points = 5,
    n_samples = 2000)
    
    
    if invert == true
        image = 1 .- image
    end
    
    ### 1. Sample points according to binomial distribution
    # for each pixel location with pixel value p, sample the point according to Binomial(1, p^2)
    sampled_img = image.^2 .> rand(Uniform(0,1), size(image))
    
    # get index of sampled points
    inds = Tuple.(findall(!iszero, sampled_img))
    points = hcat(first.(inds), last.(inds))
    
    ### 2. Remove island points 
    image_size = size(image,1)
    island_idx = find_island_idx(points, sampled_img, image_size; vicinity = vicinity, n_points = n_points) 
    
    ### 3. Downsample
    include_idx = setdiff(1:size(points,1), island_idx)

    if length(include_idx) <= n_samples
        sampled_idx = include_idx
    else 
        sampled_idx = sample(include_idx, n_samples, replace = false)
    end
    sampled_points = points[sampled_idx,:]

    # for consistency
    points = points[:,[2,1]] # needed for consistency when plotted 
    sampled_points = sampled_points[:,[2,1]] # needed for consistency when plotted 
    
    return sampled_points, points, sampled_img, island_idx
end

"""
    find_island_idx(points; <keyword arguments>)
Given sampled points from an ECM image, identify points that are "islands". That is, identify points that have less than a specified number of points in its vicinity. 
"""
function find_island_idx(points, sampled, subregion_size; vicinity = 50, n_points = 5)
    island_idx = []

    for i = 1:size(points,1)
        # look around neighborhood of size vicinity
        vicinity_half = Int(vicinity/2)
        x, y = points[i,:]

        xmin = maximum([x-vicinity_half, 1])
        xmax = minimum([x+vicinity_half, subregion_size])

        ymin = maximum([y-vicinity_half, 1])
        ymax = minimum([y+vicinity_half, subregion_size])
        sub = sampled[xmin:xmax, ymin:ymax]
        if sum(sub) < n_points
            push!(island_idx, i)
        end
    end
    return island_idx
end


function create_features_array_dim0(PI_dict, idx_ROI)
    n = length(idx_ROI)
    features_array = hcat([PI_dict[idx_ROI[i]] for i = 1:n]...)
    println("features array shape: ", size(features_array))

    features_centered = features_array .- mean(features_array, dims = 2);
    return features_array, features_centered
end

function create_features_array_dim1(PI_dict, idx_ROI)
    n = length(idx_ROI)
    features_array = hcat([vec(PI_dict[idx_ROI[i]]) for i = 1:n]...)
    println("features array shape: ", size(features_array))

    features_centered = features_array .- mean(features_array, dims = 2);
    return features_array, features_centered
end


#-----------------------------------------------------------------------------------
# functions for running persistent homology 
#-----------------------------------------------------------------------------------

"""
    run_PH(df)
Computes persistence diagrams in dimension 0 and 1 from locations

### Inputs
- `cells`: dataframe with columns `x` and `y` for (x,y) coordinates

### Outputs
- `PD0`: persistence diagram in dimension 0
- `PD1`: persistence diagram in dimension 1
"""
function run_PH(df)

    # convert to Ripser input
    P = [tuple(df[i, :x], df[i, :y]) for i = 1:size(df,1)]

    # ripser (cohomology)
    PD = ripserer(P)
    PD0 = ECM_TDA.RipsererPD_to_array(PD[1])
    PD1 = ECM_TDA.RipsererPD_to_array(PD[2])

    return PD0, PD1
end

"""
    run_PH_cell_type(cells; celltype = "cancer")
Computes persistence diagrams in dimension 0 and 1 of a specific cell type

### Inputs
- `cells`: dataframe
- `celltype` : must be one of "cancer", "leukocytes", "fibroblast"

### Outputs
- `PD0`: persistence diagram in dimension 0
- `PD1`: persistence diagram in dimension 1
"""
function run_PH_cell_type(cells; celltype = "cancer")
    
    cell_ct = cells[cells.class .== celltype, :]


    # convert to Ripser
    P = [tuple(cell_ct[i, :x], cell_ct[i, :y]) for i = 1:size(cell_ct,1)]

    # ripser (cohomology)
    PD = ripserer(P)
    PD0 = RipsererPD_to_array(PD[1])
    PD1 = RipsererPD_to_array(PD[2])

    return PD0, PD1
end

# get maximum values of persistence diagrams (for plotting purposes)
get_PD0_max(PD_dict) = maximum([maximum(PD_dict[i][1:end-1,:]) for (i,v) in PD_dict if v != reshape(Array([0.0]), 1, 1) ])
get_PD1_max(PD_dict) = maximum([maximum(PD_dict[i]) for (i,v) in PD_dict if v != reshape(Array([0.0]), 1, 1) ])

array_to_ripsererPD(PD_array) = PersistenceDiagram([(PD_array[i,1], PD_array[i,2]) for i = 1:size(PD_array,1)])


function plot_profile_dim0(idx_file, idx, ct, c, PD0, PI0, PD0_max)
    cells = CSV.read("data/4000x4000/subregion_cells/" * idx_file[idx])
    cell_ct = cells[cells.class .== ct, :]
    p1 = scatter(cell_ct.x, cell_ct.y,
                     markersize = 2,
                     yflip = true,
                     label = "",
                     markerstrokewidth = 0.2,
                     frame = :box,
                     ticks = [],
                     c = c) 
    p2 = histogram(PD0[idx][:,2], xlims = (0, PD0_max), label = "", c = "slategrey")
    p3 = heatmap(PI0[idx], xticks = [], yticks = [], legend = :none)

    return p1, p2, p3
end

function plot_profile_dim1(idx_file, idx, ct, c, PD1, PI1, PD1_max)
    cells = CSV.read("data/4000x4000/subregion_cells/" * idx_file[idx])
    cell_ct = cells[cells.class .== ct, :]
    
    p1 = scatter(cell_ct.x, cell_ct.y,
                     markersize = 2,
                     yflip = true,
                     label = "",
                     markerstrokewidth = 0.2,
                     frame = :box,
                     ticks = [],
                     c = c) 

    p2 = plot_PD(PD1[idx], pd_min = 0, pd_max = PD1_max, label = "", frame = :box, xrotation=45)
    p3 = heatmap(PI1[idx], colorbar = false, ticks = [])

    return p1, p2, p3
end

function plot_Dowker_profile(idx_file, idx, PD, PI, PD_max, 
    celltype1,
    celltype2,
    color1,
    color2)

    # get cells
    cells = CSV.read("data/4000x4000/subregion_cells/" * idx_file[idx])
    cell1 = cells[cells.class .== celltype1, :]
    cell2 = cells[cells.class .== celltype2, :]

    # plot cell 1
    p1 = scatter(cell1.x, cell1.y,
                markersize = 2,
                yflip = true,
                label = "",
                markerstrokewidth = 0.2,
                frame = :box,
                ticks = [],
                c = color1) 
    scatter!(p1, cell2.x, cell2.y,
                markersize = 2,
                yflip = true,
                label = "",
                markerstrokewidth = 0.2,
                frame = :box,
                ticks = [],
                c = color2) 
    p2 = plot_PD(PD[idx], pd_min = 0, pd_max = PD_max, label = "")
    p3 = heatmap(PI[idx], xticks = [], yticks = [], legend = :none)

    return p1, p2, p3
end




function PI1_to_PCA(PI_dict; pratio = 0.99)
    
    # subtract the mean
    n = length(PI_dict)
    PI_array = hcat([vec(PI_dict[i]) for i =1:n]...)
    PI_centered = PI_array .- mean(PI_array, dims = 2)
    
    # perform PCA
    M = fit(PCA, PI_centered, pratio = pratio)
    transformed = MultivariateStats.transform(M, PI_centered)
    
    # get eigenvectors
    n_eigenvectors = size(transformed, 1)
    eigenvectors_array = projection(M)
    eigenvectors = Dict(i => reshape(eigenvectors_array[:,i], 100,100) for i = 1:n_eigenvectors)
    
    return transformed, eigenvectors
end


function PI_to_PCA(PI_dict; pratio = 0.99)
    
    # subtract the mean
    n = length(PI_dict)
    PI_array = hcat([vec(PI_dict[i]) for i =1:n]...)
    PI_centered = PI_array .- mean(PI_array, dims = 2)
    
    # variance explained with 1 component
    M = fit(PCA, PI_centered, maxoutdim = 1)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_1 = principalratio(M)
    
    # variance explained with 2 components
    M = fit(PCA, PI_centered, maxoutdim = 2)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_2 = principalratio(M)

    # variance explained with 4 components
    M = fit(PCA, PI_centered, maxoutdim = 4)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_4 = principalratio(M)
    
    # perform PCA
    M = fit(PCA, PI_centered, pratio = pratio)
    transformed = MultivariateStats.transform(M, PI_centered)
    
    # get eigenvectors
    n_eigenvectors = size(transformed, 1)
    eigenvectors_array = projection(M)
    eigenvectors = Dict(i => reshape(eigenvectors_array[:,i], 100,100) for i = 1:n_eigenvectors)
    
    return transformed, eigenvectors, variance_1, variance_2, variance_4
end


function center_PI(PI)
    n = length(PI)
    PI_array =  hcat([vec(PI[i]) for i =1:n]...)
    PI_centered = PI_array .- mean(PI_array, dims = 2)
    return PI_centered
end

function PI0_to_PCA(PI_dict; pratio = 0.99)
    
    # subtract the mean
    n = length(PI_dict)
    PI_array = hcat([PI_dict[i] for i =1:n]...)
    PI_centered = PI_array .- mean(PI_array, dims = 2)

    # variance explained with 1 component
    M = fit(PCA, PI_centered, maxoutdim = 1)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_1 = principalratio(M)
    
    # variance explained with 2 components
    M = fit(PCA, PI_centered, maxoutdim = 2)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_2 = principalratio(M)

    # variance explained with 4 components
    M = fit(PCA, PI_centered, maxoutdim = 4)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_4 = principalratio(M)
    
    # perform PCA
    M = fit(PCA, PI_centered, pratio = pratio)
    transformed = MultivariateStats.transform(M, PI_centered)
    
    # get eigenvectors
    n_eigenvectors = size(transformed, 1)
    eigenvectors_array = projection(M)
    eigenvectors = Dict(i => eigenvectors_array[:,i] for i = 1:n_eigenvectors)
    
    return transformed, eigenvectors, variance_1, variance_2, variance_4
end


argsort(x) = reverse(sortperm(x))


function LTX_Da_idx_tuple_to_filename(LTX, Da, idx, file_idx)
    LTX_Da_idx_string = "LTX" * LTX * "_Da" * Da * "_idx" * string(idx)
    try 
        return file_idx[LTX_Da_idx_string]
    catch e
         return nothing
    end
end

function cbar_tickvals_to_loc(eigenvector_min, eigenvector_max, n, x)
    # linear function that takes eigenvector_max to n and eigenvector_min to 0
    slope = n/ (eigenvector_max - eigenvector_min)
    
    return slope * x + eigenvector_min * n /(eigenvector_min - eigenvector_max) 
end

"""
    plot_PI
plots persistence image.
### inputs
- `plot_scale`: the % of PI to plot. Defaults to 100%. 
                If provided, say X, then plot upto X% of the total rows and columns of PI matrix.
                That is, if PI is a matrix of size 100 x 100, only thow the bottom left array [1:X, 1:X]
- `x_min`, `x_max`, `y_min`, `y_max`: min, max values of the original PersistenceImage structure.
- `x_tick_interval`: size between consecutive x ticks in an interval [0, birth_max * plot_scale]
- `y_tick_interval`: size between consecutive y ticks in an interval [0, persistence_max * plot_scale]

### outputs
"""
function plot_PI(PI, x_min, x_max, y_min, y_max; 
                plot_scale = 100,
                x_tick_interval = 400,
                y_tick_interval = 400,
                kwargs...) 
    
    n = size(PI,1)
    
    # scale all min, max
    x_min_scaled = x_min * plot_scale / 100
    x_max_scaled = x_max * plot_scale / 100
    y_min_scaled = y_min * plot_scale / 100
    y_max_scaled = y_max * plot_scale / 100
    
    # locate "0"
    x_loc0 = 0 - x_min
    y_loc0 = 0 - y_min
    
    x_zero_tick = x_loc0/ (x_max - x_min) * n
    y_zero_tick = y_loc0 / (y_max - y_min) * n
    
    # compute the number of ticks to draw
    x_tick_loc = x_tick_interval/x_max_scaled * plot_scale 
    y_tick_loc = y_tick_interval/y_max_scaled * plot_scale 

    p = heatmap(PI[1:plot_scale, 1:plot_scale], 
        label = "",
        title = "",
        framestyle = :box,
        xticks = (x_zero_tick:x_tick_loc:plot_scale, Int32.(round.(0:x_tick_interval:x_max_scaled))),  # (location of ticks, tick values)
        yticks = (y_zero_tick:y_tick_loc:plot_scale, Int32.(round.(0:y_tick_interval:y_max_scaled)))
        ;kwargs...)
   
    return p
end


function plot_scores(scores; n_coordinates = nothing, limit = nothing, xtickfontsize = 12, ytickfontsize = 10, 
                    coord_label = "coordinate ",kwargs...)

    score_min = minimum(scores)
    score_max = maximum(scores)
    if limit == nothing
        limit = maximum([abs(score_min), abs(score_max)]) * 1.1
    end
    
    if n_coordinates != nothing
        scores = scores[1:n_coordinates]
        n = n_coordinates
    else
        n = length(scores)
    end

    xticks = (collect(1:n), [coord_label * string(i) for i = 1:n])
    

    p = Plots.bar(scores, 
                label = "", 
                xticks= xticks,
                xtickfontsize = xtickfontsize,
                ytickfontsize = ytickfontsize,
                #yticks = 3,
                frame = :box,
                color = ifelse.(scores .< 0, "#e56b6f", "#355070"),
                ylims = (-limit, limit),
                kwargs...
                )
    hline!(p, [0,0], label = "", c = "black")
    return p
end

function get_large_coordinate_examples(transformed, i, thresholds; n_coordinates = nothing)
    if n_coordinates == nothing
        n = size(transformed,1)
    else
        n = n_coordinates
    end
        
    example_idx = reverse(sortperm(transformed[i,:]))

    # select indices only if i is the largest coordinate
    example_idx = [k for k in example_idx if i == findmax(transformed[:,k])[2]]
    
    # select indices only if the remaining coordinates are small enough
    selected = []
    other_coords = [j for j=1:n if j != i]
    for k in example_idx
        append_bool = true
        for j in other_coords
            if transformed[j,k] > thresholds[j]
                append_bool = false
            end
        end
        
        if append_bool == true
            append!(selected, k) 
        end
    end
    return selected
    
end

function find_idx_from_range(range, x)
    for (idx, i) in enumerate(range)
        if i > x
           return idx-1 
        end
    end
end
"""
    sample_uniform(cell_df; <keyword arguments>)
Given a dataframe of cells, sample uniformly without replacement.
"""
function sample_uniform(cell_df; subsample_size = 400)
    n = size(cell_df, 1)
    if n > subsample_size
        sampled_idx = sample(1:size(cell_df, 1), subsample_size, replace = false)
        return cell_df[sampled_idx, :]  
    else
        return cell_df
    end
end

"""
    sample_kde(cell_df; <keyword arguments>)
Given a dataframe of cells, estimate the KDE and sample (without replacement) from KDE
"""
function sample_kde(cell_df; subsample_size = 400)
    n = size(cell_df, 1)

    if n > subsample_size
        # perform KDE
        P = convert(Array,cell_df[:, [:x, :y]]);
        P_kde = kde(P)
        
        # for each original point, get KDE values
        kde_val = []
        for i =1:size(cell_df, 1)
            # find index of x and y in the grid used in KDE
            x_idx = find_idx_from_range(P_kde.x, cell_df[i, :x])
            y_idx = find_idx_from_range(P_kde.y, cell_df[i, :y])

            append!(kde_val, P_kde.density[x_idx, y_idx])
        end

        sampled_idx = sample(1:size(cell_df, 1), Weights(Float64.(kde_val)), subsample_size, replace = false);
        return cell_df[sampled_idx, :], P_kde
    else
        return cell_df, nothing
    end 
end

"""compute_Dowker();

### Inputs
- `P1`: Array of size (n, 2)
- `P2`: Array of size (m, 2)
### Outputs
"""
function compute_Dowker(P1, P2, min_size = 20)
    
    P1_size = size(P1, 1)
    P2_size = size(P2, 1)
    
    if (P1_size > min_size) & (P2_size > min_size)

        # comute pairwise distances between P1 and P2
        d = Distances.pairwise(Euclidean(), P1, P2, dims = 1)

        if P1_size > P2_size
            d = Array(transpose(d)) 
        end

        # run Witness persistence
        W = ECM_TDA.compute_Witness_persistence(d)

        W_barcode0 = Eirene.barcode(W["eirene_output"], dim = 0) 
        W_barcode1 = Eirene.barcode(W["eirene_output"], dim = 1) 

        return W_barcode0, W_barcode1, d, W
    else
        
       return nothing, nothing, nothing, nothing
    end
end

"""compute_Dowker_cells();

"""
function compute_Dowker_cells(cells;
        celltype1 = "cancer",
        celltype2 = "leukocytes",
        subsample_size = 300,
        min_cellsize = 20)
    
    
    cells1 = convert(Matrix, cells[cells.class .== celltype1,[:x, :y]])
    cells2 = convert(Matrix, cells[cells.class .== celltype2,[:x, :y]])

    cell1_size = size(cells1,1)
    cell2_size = size(cells2,1)

    if (cell1_size > min_cellsize) & (cell2_size > min_cellsize)
        if subsample_size >0

            if cell1_size > subsample_size
                cell1_idx = sample(1:cell1_size, subsample_size, replace = false)
                cells1 = cells1[cell1_idx,:]
            end


            if cell2_size > subsample_size
                cell2_idx = sample(1:cell2_size, subsample_size, replace = false)
                cells2 = cells2[cell2_idx,:]
            end
        end

        ### Compute Witness persistence  
        # comute pairwise distances between leukocytes & ecm

        d = Distances.pairwise(Euclidean(), cells1, cells2, dims = 1);

        # run Witness persistence
        W = compute_Witness_persistence(d)

        W_barcode0 = Eirene.barcode(W["eirene_output"], dim = 0) 
        W_barcode1 = Eirene.barcode(W["eirene_output"], dim = 1) 

        return W_barcode0, W_barcode1, cells1, cells2, d
    else
        
       return nothing, nothing, nothing, nothing, nothing 
    end
end


"""load_ECM_data_v1v2v3()
Load all ECM images. 
***For images in data_v2 and data_v3, we halve their size due to indexing inconsistencies***
function load_ECM_data_v1v2v3(v1_LTX_Da, v2_LTX_Da, v3_LTX_Da)
    
    # load LTX Da dictionary
    #LTX_Da = load("LTX_Da_dict.jld2")
    #v1_LTX_Da = LTX_Da["v1_LTX_Da"]
    #v2_LTX_Da = LTX_Da["v2_LTX_Da"]
    #v3_LTX_Da = LTX_Da["v3_LTX_Da"]
    
    # specify file prefix
    v1_csv_pre = "data_v1/region_csv/"
    v1_deconvolved_pre = "data_v1/outputs_stitched/";
    v2_csv_pre = "data_v2/region_csv/"
    v2_deconvolved_pre = "data_v2/outputs_stitched_deconvolution/"
    v3_csv_pre = "data_v3/region_csv/"
    v3_deconvolved_pre = "data_v3/stitched_deconvolved_PSR/";
    
    img_dict = Dict()

    # load images from v1
    for LTX in keys(v1_LTX_Da)
        for Da in v1_LTX_Da[LTX]
            img_path = get_image_path(v1_deconvolved_pre, LTX, Da)
            img = Images.load(img_path)
            img = 1 .- img
            img_dict[(LTX, Da)] = img
            #print("\nLTX: ", LTX, " Da: ", Da, " size: ", size(img))
        end
    end

    # load images from v2
    for LTX in keys(v2_LTX_Da)
        for Da in v2_LTX_Da[LTX]
            img_path = get_image_path(v2_deconvolved_pre, LTX, Da)
            img = Images.load(img_path)
            img = 1 .- img

            # halve the size
            new_size = trunc.(Int, size(img) .* 0.5)
            img = imresize(img, new_size)

            img_dict[(LTX, Da)] = img
            #print("\nLTX: ", LTX, " Da: ", Da, " size: ", size(img))
        end
    end

    # load images from v3
    for LTX in keys(v3_LTX_Da)
        for Da in v3_LTX_Da[LTX]
            img_path = get_image_path_v3(v3_deconvolved_pre, LTX, Da)
            img = Images.load(img_path)
            img = 1 .- img

            # halve the size
            new_size = trunc.(Int, size(img) .* 0.5)
            img = imresize(img, new_size)

            img_dict[(LTX, Da)] = img
            #print("\nLTX: ", LTX, " Da: ", Da, " size: ", size(img))
        end
    end
    
    return img_dict
end

function load_cell_locations(v1_LTX_Da, v2_LTX_Da, v3_LTX_Da)
    # prefix
    v1_csv_pre = "data_v1/region_csv/"
    v2_csv_pre = "data_v2/region_csv/"
    v3_csv_pre = "data_v3/region_csv/"
    
    # load cell location info
    cell_dict = Dict()

    for LTX in keys(v1_LTX_Da)
        for Da in v1_LTX_Da[LTX]
            csv_path = get_csv_path(v1_csv_pre, LTX, Da)
            locations = CSV.read(csv_path)
            cell_dict[(LTX, Da)] = locations
        end
    end

    for LTX in keys(v2_LTX_Da)
        for Da in v2_LTX_Da[LTX]
            csv_path = get_csv_path(v2_csv_pre, LTX, Da)
            locations = CSV.read(csv_path)
            cell_dict[(LTX, Da)] = locations
        end
    end

    for LTX in keys(v3_LTX_Da)
        for Da in v3_LTX_Da[LTX]
            csv_path = get_csv_path(v3_csv_pre, LTX, Da)
            locations = CSV.read(csv_path)
            cell_dict[(LTX, Da)] = locations
        end
    end
    return cell_dict
end
"""

function get_subregion_boundaries(center_x, center_y, subregion_size)
    subregion_half = Int(subregion_size // 2)
    xmin = center_x - subregion_half
    xmax = center_x + subregion_half
    ymin = center_y - subregion_half
    ymax = center_y + subregion_half
    
    return xmin, xmax, ymin, ymax
    
end



function get_subimage_dict(img_dict, subregion_centers, subregion_size; flip_xy = false)
    sub_img = Dict()
    for (LTX, Da) in keys(img_dict)
        for (idx, center) in enumerate(subregion_centers[(LTX, Da)])
            img = img_dict[(LTX, Da)]
            
            if flip_xy == false
                img_sub = get_img_sub(img, center, subregion_size)
            elseif flip_xy == true
                img_sub = get_img_sub(img, [center[2], center[1]], subregion_size)
            end
                
                
            sub_img[(LTX, Da, idx)] = img_sub
        end

    end
    return sub_img
end



function get_img_sub(img, center, subregion_size)
    center_x, center_y = center
    xleft = center_y - Int(subregion_size // 2) + 1
    xright = center_y + Int(subregion_size // 2) 
    yleft = center_x - Int(subregion_size // 2) + 1
    yright = center_x + Int(subregion_size // 2)
    img_sub = img[yleft:yright, xleft:xright]  
    return img_sub
end



function plot_subregions(image, subregion_centers, subregion_size; lw= 5, kwargs...)
    p = plot(image, xaxis = :false, yaxis = :false, background_color = :transparent; kwargs...)
    for (idx, center) in enumerate(subregion_centers)
        center_x, center_y = center
        xleft = center_y - Int(subregion_size // 2)
        xright = center_y + Int(subregion_size // 2)
        yleft = center_x - Int(subregion_size // 2)
        yright = center_x + Int(subregion_size // 2)
        plot!(p, [xleft, xright], [yleft, yleft], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xleft, xleft], [yleft, yright], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xright, xright], [yleft, yright], linewidth = lw, color = :deeppink, label = "")
        plot!(p, [xleft, xright], [yright, yright], linewidth = lw, color = :deeppink, label = "") 
        scatter!(p, [center_y], [center_x], text = string(idx), label = "", alpha = 0)
    end
    return p
end

function plot_subregion_boundaries(img_dict, cell_dict, LTX, Da, celltypes, celltype_colors, subregion_centers, subregion_size)
    p = plot_ROI_ECM_cells(img_dict, cell_dict, LTX, Da, celltypes, celltype_colors)
    for j in 1:size(subregion_centers[(LTX, Da)],1)
        x, y = subregion_centers[(LTX, Da)][j]
        subregion_half = Int(subregion_size // 2) 
        xmin = x - subregion_half
        xmax = x + subregion_half
        ymin = y - subregion_half
        ymax = y + subregion_half
        plot!(p,[xmin, xmax], [ymin, ymin], linewidth = 5, color = :deeppink, label = "")
        plot!(p,[xmin, xmin], [ymin, ymax], linewidth = 5, color = :deeppink, label = "")
        plot!(p,[xmax, xmax], [ymin, ymax], linewidth = 5, color = :deeppink, label = "")
        plot!(p,[xmin, xmax], [ymax, ymax], linewidth = 5, color = :deeppink, label = "")
        scatter!(p, [x], [y], text = string(j), label = "", alpha = 0)

    end
    return p
    
end

function plot_ROI_ECM_cells(img_dict, cell_dict, LTX, Dacell_types, celltype_colors)
    # get ECM image and cell locations
    ROI_img = img_dict[(LTX, Da)]
    locations = cell_dict[(LTX, Da)]

    # plot ECM image and leukocyte location
    p = plot(plot(Gray.(ROI_img)), 
            framestyle = :box,
            xticks = false,
            yticks = false)

    for ct in cell_types
        # get locations of cell type
        location_ct = locations[locations.class .== ct, :]
        c = celltype_colors[ct]
        # get color
        scatter!(p, location_ct.x, location_ct.y,
                 markersize = 2,
                 yflip = true,
                 label = ct,
                 markerstrokewidth = 0.2,
                 c = c) 

    end

    return p
end


"""
    get_tile_centers(ROI_size; tilesize = 4000)
Gets tile center for non-overlapping tiles
"""
function get_tile_centers(ROI_size; tilesize = 4000)
    subregion_centers = Dict()
    
    for (key, val) in ROI_size
        size_x, size_y = val
        tile_x = [Int(tilesize/2) + i * tilesize for i = 0:(size_x/tilesize)-1]
        tile_y = [Int(tilesize/2) + i * tilesize for i = 0:(size_y/tilesize)-1]
        subregion_centers[key] = [(Int(i),Int(j)) for i in tile_x for j in tile_y]
    end
    return subregion_centers
end






function run_PH_whole_image(path_dict; p = 0.99)
    PH = Dict()
    
    for (i, path) in path_dict
        # load image
        data = Images.load(path)
        
        # invert
        data_inv = 1 .- data
        
        # sample points
        points, _ = grayscale_to_pointcloud2(data_inv; p = p)
        
        # convert to Ripser format
        P = [tuple(points[1,i], points[2,i]) for i =1:size(points,2)]
        
        # ripser (cohomology)
        PD = ripserer(P)
        
        PH[i] = PD
    end
    
    return PH
    
end

# convert to 2D arrays
function RipsererPD_to_array(PD)
    n = size(PD,1)
    PD_array = zeros(n, 2)
    for i = 1:n
        PD_array[i,:] .= PD[i]
    end
    return PD_array
end

function get_ranked_persistence(PD_dict)
    persistence_dict = Dict()
    for (k,pd) in PD_dict
        persistence = sort(pd[:,2] - pd[:,1], rev = true)
        persistence = [x for x in persistence if x != Inf]
        persistence_dict[k] = persistence
    end

    # compute the min length of persistence
    min_length = minimum([length(v) for (k,v) in persistence_dict])
    
    # truncate the persistence vector
    persistence_dict = Dict(k => v[1:min_length] for (k,v) in persistence_dict)

    return persistence_dict, min_length
end

"""
    
"""
function compute_PI(PH_dict; sigma = 50, size = 20)
    
    PI = PersistenceImage([PH_dict[k] for k in keys(PH_dict)], sigma=sigma, size = size)
    PH_PI = Dict()
    for i in keys(PH_dict)
        PH_PI[i] = PI(PH_dict[i])
    end
    return PH_PI
end

function plot_barcode(barcode; 
    color = :grey56, # default bar color
    return_perm = false, # whether to return the permutation index or not
    kwargs...)

    # adjust linewidth, depending on number of intervals
    n = size(barcode,1)

    # find ordering according to birth time
    perm = sortperm(barcode[:,1])

    # non-inf maximum death time
    if filter(!isinf,barcode[:,2]) != []
        death_max = maximum(filter(!isinf,barcode[:,2])) 
    else
        death_max = maximum(barcode[:,1]) * 2
    end

    p = plot(framestyle = :box,
            top_margin = 5 * Plots.PlotMeasures.mm, 
            bottom_margin = 5 * Plots.PlotMeasures.mm, 
            yaxis = nothing;
            kwargs...)
    
    # plot all bars
    idx = 1
    for i in perm
        birth = barcode[i,1]
        death = barcode[i,2]
        
        # assign a death time if bar has infinite death time 
        if isinf(death)
            death = death_max * 1.2
        end
        
        plot!(p,[birth,death],[idx,idx], legend = false, linecolor = color, hover = "class " *string(i); kwargs...)
        idx += 1
    end

    ylims!((-1, n+1))
    
    if return_perm == true
        return p, perm
    else
        return p
    end
end

function plot_PD(barcode; 
    highlight = [], highlight_color = :deeppink2, cutoff = nothing, inf_coord = nothing,
    pd_min = nothing, pd_max = nothing, threshold_lw = 5, diagonal_lw = 5, inf_markerstrokewidth = 5,
    kwargs...)
points = barcode


    if size(barcode,1) == 0
        # plot diagonal line
        p = plot([0, 1], [0, 1], 
        labels ="", 
        linewidth = diagonal_lw,
        framestyle = :box,
        xlims = (0,1),
        ylims = (0,1),
        aspect_ratio = :equal,
        color = "grey"; 
        kwargs...)
        return p
    end
    # find index of points with death parameter == death
    idx = findall(x -> x == Inf, points[:,2])

    # plot points with death < Inf
    idx2 = [i for i in 1:size(points,1) if i ∉ idx]
    p = scatter(points[idx2,1], points[idx2,2]; kwargs..., color = "grey", labels = "", alpha = 0.5)

    # find max death value
    max_death = maximum(points[idx2, 2])

    # plot points with death parameter == Inf
    if inf_coord == nothing
        death_plot = ones(size(idx,1)) * max_death
    else
        death_plot = ones(size(idx, 1)) * inf_coord
    end
    scatter!(p, points[idx,1], death_plot, marker = :xcross, 
            markerstrokewidth = inf_markerstrokewidth,
            aspect_ratio = :equal, legend=:bottomright, labels="", color ="red"; kwargs...)

    # plot diagonal line
    if pd_max == nothing
        
        min_birth = minimum(points[:,1]) * 0.8
        max_death = max_death * 1.1
        plot!(p, [min_birth, max_death], [min_birth, max_death], 
            labels ="", 
            linewidth = diagonal_lw,
            framestyle = :box,
            xlims = (min_birth, max_death),
            ylims = (min_birth, max_death),
            color = "grey"; 
            kwargs...)
    else
        max_death = pd_max
        min_birth = pd_min
        plot!(p, [min_birth, max_death], [min_birth, max_death], 
            labels ="", 
            linewidth = diagonal_lw,
            framestyle = :box,
            xlims = (min_birth, max_death),
            ylims = (min_birth, max_death),
            color = "grey"; 
            kwargs...)
    end

    # if highlight is provided, color specific points with the given color
    if highlight != []
        scatter!(points[highlight,1], points[highlight,2]; kwargs..., color = highlight_color, labels = "", hover = highlight)
    end

    # plot the cutoff (dotted line) if provided
    if cutoff != nothing
        f(x) = x + cutoff
        plot!(f, linestyle = :dash, c = "black", label = "", hover = false, linewidth = threshold_lw)
    end

    return p
end

"""
    plot_cyclerep()
Plots a single point cloud P in 2-dimensions and a 1-dimensional cycle. 
"""
function plot_cyclerep(P; cycle = [], cycle_color = "deeppink1", cycle_lw = 5, kwargs...)
    # P: array of size (2, n) or (3,n)
    # cycle: classrep returned by Eirene
    
    # plot points P
    p = plot(P[1,:], P[2,:], 
            seriestype = :scatter, 
            label = "",
            framestyle = :box,
            xaxis = nothing,
            yaxis = nothing;
            kwargs...)
    
    # plot cycle
    cycle_v = unique(cycle)
    
    scatter!(P[1,cycle_v], P[2, cycle_v], color = cycle_color, label ="")
    
    return p
end

function compute_PI_dissimilarity(PI_dict)
# PI_dict must be a dictionary where keys have index 1, 2, ... and the values are the persistence images
    
    n = length(PI_dict)
    D = zeros((n, n))
    for i = 1:n
        for j = i+1:n
            D[i,j] = D[j,i] = norm(PI_dict[i] - PI_dict[j])
        end
    end
    return D
    
end

"""
    index_closest_to_x
Given a list `l`, find index of element closes to `x`
"""
function index_closest_to_x(l, x)
    d = [abs(i - x) for i in l]
    idx = argmin(d)
    return idx
end

function get_1simplices(D_param)
    n_cols = size(D_param, 2)
    one_simplices = []
    for i = 1:n_cols
        rows = findall(x -> x == 1, D_param[:,i])
        append!(one_simplices, combinations(rows, 2))
    end
    
    return one_simplices
end

function get_2simplices(D_param)
    n_cols = size(D_param,2)
    two_simplices = []
    for i = 1:n_cols
        ones = findall(D_param[:,i])
        append!(two_simplices, collect(combinations(ones,3)))
    end
    
    return unique(two_simplices)
end

"""
    plot_Dowker_complex
Given a dissimilarity matrix D, compute the Dowker complex at parameter `param`.
Uses the rows as potential vertex set. Must provide `PC`, the locations of vertex set corresponding to the rows.

### Inputs
- `D`: Dissimilarity matrix
- `param`: Parameter. Builds the Dowker complex at this parameter.
- `PC`: An array of size (n,2), where n is the number of rows of D. 
        Provides (x,y)-coordinates of vertices corresponding to the  rows of D. 

### Outputs
- a plot
"""
function plot_Dowker_complex(D, param, PC; 
                             show_2simplex = false, 
                             show_unborn_vertices = false,
                             c = "#29A0B1",
                            kwargs...)
    n_rows, n_cols = size(D)
    D_param = D .< param
    
    p = plot()
    
    # plot 2-simplices
    if show_2simplex == true
        two_simplices = get_2simplices(D_param)
        for simplex in two_simplices
            plot!(p, Plots.Shape([(PC[i,1], PC[i,2]) for i in simplex]), 
                 label="", legend = :false, c = c, alpha = 0.1
                 )
        end
    end
    
    # plot 1-simplices
    one_simplices = get_1simplices(D_param)
    for simplex in one_simplices
        plot!(p, [PC[simplex[1],1], PC[simplex[2],1]], [PC[simplex[1],2], PC[simplex[2],2]], label = "", c = :black) 
    end
    
    # plot 0-simplices
    idx0 = findall(x -> x != 0, vec(sum(D_param, dims = 2)))
    scatter!(p, PC[idx0,1], PC[idx0, 2]; label = "", frame = :box, ticks = [], c = c, aspect_ratio = :equal, kwargs...)
    
    # plot unborn vertices 
    if show_unborn_vertices == true
        idx_unborn = findall(x -> x == 0, vec(sum(D_param, dims = 2)))
        scatter!(p, PC[idx_unborn,1], PC[idx_unborn, 2], label = "",
                 markershape = :xcross,
                 markerstrokewidth = 3,
                 c = c)
        
    end
    
    
    return p
    
end

function plot_dim_red(y, groups; 
    dim_red = "UMAP", 
    xaxis = "UMAP-1", 
    yaxis = "UMAP-2", 
    kwargs...)
    """
    groups: Dictionary of form (i => [indices])
    """

    ### specify colors ###
    c = Dict(
    0 => :grey,
    1 => "#fd5184", # pink
    2=> "#F28522", # orange
    3 => "#ffb602", # yellow
    4 => "#AEF359", # lime green
    5 => "#49a849", # green
    6 => "#3ec0c9", # blue / teal 
    7 => "#265BF5", # dark blue 
    8 => "#8B008B", # purple 
    9 => "#215467", # ocean
    )

    ### marker shapes ###
    markershapes = Dict(1 => :rect,
            2 => :utriangle,
            3 => :star,
            4 => :pentagon,
            5 => :diamond,
            6 => :dtriangle,
            7 => :star8,
            8 => :octagon,
            9 => :star4,
            10 => :pentagon
            )
    ### label
    if dim_red == "UMAP"
        label_prefix = "U"
    else
        label_prefix = "R"
    end

    # other figure parameters
    markersize = 3
    legendfontsize = 5
    markerstrokewidth = 1

    n = size(y, 2)
    n_groups = length(groups)

    annotated = vcat([val for (key, val) in groups]...)
    nonannotated = [i for i = 1:size(y, 2) if i ∉ annotated]
    p = scatter(y[1,nonannotated], y[2,nonannotated];
                markercolor = "lightgray",
                alpha = 0.6,
                markersize = markersize, 
                markerstrokewidth = markerstrokewidth,
                label = "", 
                xaxis = xaxis,
                yaxis = yaxis,
                ticks = [],
                guidefontsize = 7,
                framestyle = :box,
                size = (200, 150),
                background_color=:transparent, foreground_color=:black,
                kwargs...
    )


    for k = 1:n_groups
        v = groups[k]
        scatter!(y[1,v], y[2,v], 
                markersize = markersize,
                markerstrokewidth = markerstrokewidth,
                markershape = markershapes[k],
                c = c[k],
                labels = label_prefix * string(k),
                legendfontsize = legendfontsize
                )     
    end
    return p

end



# get indices with large and small i-th coordinates
function get_coordinate_min_max_examples(transformed, i; n =4)

    sorted = sortperm(transformed[i,:])
    min_indices = sorted[1:n]
    max_indices = sorted[end-n+1:end]
    return min_indices, max_indices
end

function plot_low_high_PC_cancer_PSRH(min_indices, max_indices, idx_ROI, save_filename)
    gr()
    plot_array = []
    n = length(min_indices)
    
    for indices in [max_indices, min_indices]
        for idx in indices
            f = idx_ROI[idx]
        
            # cancer
            df = CSV.read("data/4000x4000_combined/subregion_cells/" * f * ".csv")
            df_cell = df[df.class .== "cancer", :]
            p_C = scatter(df_cell.x, df_cell.y,
                             markersize = 1.5,
                             yflip = true,
                             label = "",
                             markerstrokewidth = 0.2,
                             frame = :box,
                             ticks = [],
                            aspect_ratio = :equal,
                            size = (150,150),
                             c = c_cancer,
                             right_margin = -4mm) 
            push!(plot_array, p_C)

            # leukocytes
            df = CSV.read("data/4000x4000_combined/subregion_cells/" * f * ".csv")
            df_cell = df[df.class .== "leukocytes", :]
            p_L = scatter(df_cell.x, df_cell.y,
                             markersize = 1.5,
                             yflip = true,
                             label = "",
                             markerstrokewidth = 0.2,
                             frame = :box,
                             ticks = [],
                            aspect_ratio = :equal,
                            size = (150,150),
                             c = c_leukocytes,
                             right_margin = -4mm) 
            push!(plot_array, p_L)
        


            p_PSRH = Images.load("data/4000x4000_combined/PSRH/" * f * ".tif")
            push!(plot_array, plot(p_PSRH, ticks = [], frame = :box, bottom_margin = -3mm, right_margin = 5mm))
        end
    end
    p = plot(plot_array..., layout = grid(2, n * 2), size = (250 * n * 2, 250 * 2))
    savefig(save_filename)   
end

function plot_low_high_PC_cancer_leukocytes_PSRH(min_indices, max_indices, idx_files, save_filename)
    gr()
    plot_array = []
    n = length(min_indices)
    
    for indices in [max_indices, min_indices]
        for idx in indices
            f = idx_files[idx]
        
             
            # get cells
            df = CSV.read("data/4000x4000_combined/subregion_cells/" * idx_files[idx] * ".csv")

            # plot cancer
            df_cell = df[df.class .== "cancer", :]
            p = scatter(df_cell.x, df_cell.y,
                             markersize = 1.5,
                             yflip = true,
                             label = "",
                             markerstrokewidth = 0.2,
                             frame = :box,
                             ticks = [],
                            aspect_ratio = :equal,
                            size = (150,150),
                            background_color=:transparent, foreground_color=:black, 
                             c = c_cancer,
                            right_margin = -5mm) 

            push!(plot_array, p)

            
            # plot cancer
            df_cell = df[df.class .== "leukocytes", :]
            p = scatter(df_cell.x, df_cell.y,
                             markersize = 1.5,
                             yflip = true,
                             label = "",
                             markerstrokewidth = 0.2,
                             frame = :box,
                             ticks = [],
                            aspect_ratio = :equal,
                            size = (150,150),
                            background_color=:transparent, foreground_color=:black, 
                             c = c_leukocytes,
                            right_margin = -5mm) 

            push!(plot_array, p)


            p_PSRH = Images.load("data/4000x4000_combined/PSRH/" * f * ".tif")
            push!(plot_array, plot(p_PSRH, ticks = [], frame = :box, bottom_margin = -3mm, right_margin = 5mm))
        end
    end
    p = plot(plot_array..., layout = grid(2, n * 3), size = (250 * n * 3, 250 * 2))
    savefig(save_filename)   
end

function get_small_large_coordinate_examples(transformed, i, thresholds; n_coordinates = nothing, lim = 4)
    if n_coordinates == nothing
        n = size(transformed,1)
    else
        n = n_coordinates
    end
    
    
    ### get index of large coordinates
    example_idx = reverse(sortperm(transformed[i,:]))

    # select indices only if i is the largest coordinate
    example_idx = [k for k in example_idx if i == findmax(transformed[:,k])[2]]

    # select indices only if the remaining coordinates are small enough
    large_idx = []
    other_coords = [j for j=1:n if j != i]
    for k in example_idx
        append_bool = true
        for j in other_coords
            if abs(transformed[j,k]) > thresholds[j]
                append_bool = false
            end
        end
        
        if append_bool == true
            append!(large_idx, k) 
        end
    end
    
    ### get index of small coordinates
    example_idx = sortperm(transformed[i,:])

    # select indices only if i is the largest coordinate
    example_idx = [k for k in example_idx if i == findmin(transformed[:,k])[2]]
    
    # select indices only if the remaining coordinates are small enough
    small_idx = []
    other_coords = [j for j=1:n if j != i]
    for k in example_idx
        append_bool = true
        for j in other_coords
            if abs(transformed[j,k]) > thresholds[j]
                append_bool = false
            end
        end
        
        if append_bool == true
            append!(small_idx, k) 
        end
    end
    
    return small_idx[1:lim], large_idx[1:lim]
end

function plot_Dowker_profile_cells(
    nonempty_keys_to_original, 
    idx_file, 
    idx, 
    PD, 
    PI, 
    PD_max, 
    celltype1,
    celltype2,
    color1,
    color2)

    # get original cells
    idx_new = nonempty_keys_to_original[idx]
    cells = CSV.read("data/4000x4000/subregion_cells/" * idx_file[idx_new])
    cell1 = cells[cells.class .== celltype1, :]
    cell2 = cells[cells.class .== celltype2, :]

    # plot cell 1
    p1 = scatter(cell1.x, cell1.y,
                markersize = 2,
                yflip = true,
                label = "",
                markerstrokewidth = 0.2,
                frame = :box,
                ticks = [],
                c = color1) 
    p2 = scatter(cell2.x, cell2.y,
                markersize = 2,
                yflip = true,
                label = "",
                markerstrokewidth = 0.2,
                frame = :box,
                ticks = [],
                c = color2) 
    
    # load sampled cells
    filename = idx_file[idx_new]
    sampled_cells1 = readdlm("data/4000x4000/cells_sampled/kde_sample/"  * celltype1 * "/" * filename, ',')
    sampled_cells2 = readdlm("data/4000x4000/cells_sampled/kde_sample/" * celltype2 * "/" * filename, ',');
    
    p3 = scatter(sampled_cells1[:,1], sampled_cells1[:,2], ticks = [], frame = :box, label = "", yflip = :true, c = color1, markersize = 2, markerstrokewidth = 0.2)
    p4 = scatter(sampled_cells2[:,1], sampled_cells2[:,2], ticks = [], frame = :box, label = "", yflip = :true, c = color2, markersize = 2, markerstrokewidth = 0.2)
    p5 = ECM_TDA.plot_PD(PD[idx_new], pd_min = 0, pd_max = PD_max, label = "")
    p6 = heatmap(PI[idx_new], xticks = [], yticks = [], legend = :none)

    return p1, p2, p3, p4, p5, p6
end

# recompute PI (lower resolution)
function compute_PI2(PD; sigma = 50, size = 20)
    PH_dict = Dict(k => ECM_TDA.array_to_ripsererPD(v) for (k,v) in PD if v != nothing);

    PI = PersistenceImage([PH_dict[k] for k in keys(PH_dict)], sigma=sigma, size = size)
    PH_PI = Dict()
    for i in keys(PH_dict)
        PH_PI[i] = PI(PH_dict[i])
    end
    return PH_PI
end

function combine_PI0_PI1_dicts_Dowker(PI0, PI1)
    # select dictionaries
    dicts = [PI0,
            PI1,
            ];

    # get keys that are present in all dictionaries
    all_keys = []
    for k in keys(dicts[1])
        present = 0
        for j = 2:length(dicts)
            if k in keys(dicts[j])
                present += 1
            end
        end

        if present == length(dicts) - 1
            push!(all_keys, k)
        end
    end

    # combine all features
    features = Dict()
    for f in all_keys
        combined = vcat(
                    vec(PI0[f]), 
                    vec(PI1[f])
                    )
        features[f] = combined
    end

    return features
end

function centered_features_to_PCA(features_centered; pratio = 0.99)
    
    # variance explained with 1 component
    M = fit(PCA, features_centered, maxoutdim = 1)
    transformed = MultivariateStats.transform(M, features_centered)
    variance_1 = principalratio(M)
    
    # variance explained with 2 components
    M = fit(PCA, features_centered, maxoutdim = 2)
    transformed = MultivariateStats.transform(M, features_centered)
    variance_2 = principalratio(M)

    # variance explained with 4 components
    M = fit(PCA, features_centered, maxoutdim = 4)
    transformed = MultivariateStats.transform(M, features_centered)
    variance_4 = principalratio(M)
    
    # perform PCA
    M = fit(PCA, features_centered, pratio = pratio)
    transformed = MultivariateStats.transform(M, features_centered)

   
    return transformed, variance_1, variance_2, variance_4
end

function PI_to_PCA2(PI_dict; pratio = 0.99)
    
    # subtract the mean
    n = length(PI_dict)
    PI_array = hcat([vec(PI_dict[i]) for i =1:n]...)
    PI_centered = PI_array .- mean(PI_array, dims = 2)
    
    # variance explained with 1 component
    M = fit(PCA, PI_centered, maxoutdim = 1)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_1 = principalratio(M)
    
    # variance explained with 2 components
    M = fit(PCA, PI_centered, maxoutdim = 2)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_2 = principalratio(M)

    # variance explained with 4 components
    M = fit(PCA, PI_centered, maxoutdim = 4)
    transformed = MultivariateStats.transform(M, PI_centered)
    variance_4 = principalratio(M)
    
    # perform PCA
    M = fit(PCA, PI_centered, pratio = pratio)
    transformed = MultivariateStats.transform(M, PI_centered)
    
    # get eigenvectors
    n_eigenvectors = size(transformed, 1)
    eigenvectors_array = projection(M)
    eigenvectors = Dict(i => reshape(eigenvectors_array[:,i], 20,20) for i = 1:n_eigenvectors)
    
    return transformed, eigenvectors, variance_1, variance_2, variance_4
end

function plot_PI2_old(PI, x_min, x_max, y_min, y_max; 
    x_tick_interval = 400,
    y_tick_interval = 400,
    kwargs...) 

    n = size(PI,1)
    xinterval = (x_max - x_min)/4
    yinterval = (y_max - y_min)/4

    p = heatmap(PI, 
    label = "",
    title = "",
    framestyle = :box,
    xticks = (5:5:20, Int32.(round.(x_min+xinterval:xinterval:x_max))),
    yticks = (5:5:20, Int32.(round.(y_min+yinterval:yinterval:y_max))),
    ;kwargs...)

    return p
end

function plot_PI2(PI, x_min, x_max, y_min, y_max; 
    x_tick_interval = 400,
    y_tick_interval = 400,
    kwargs...) 

    n = size(PI,1)

    # locate "0"
    x_loc0 = 0 - x_min
    y_loc0 = 0 - y_min
    
    x_zero_tick = x_loc0/ (x_max - x_min) * n
    y_zero_tick = y_loc0 / (y_max - y_min) * n
    
    # compute the number of ticks to draw
    x_tick_loc = x_tick_interval/x_max *n
    y_tick_loc = y_tick_interval/y_max *n

    p = heatmap(PI, 
    label = "",
    title = "",
    framestyle = :box,
    xticks = (x_zero_tick:x_tick_loc:n, Int32.(round.(0:x_tick_interval:x_max))),  # (location of ticks, tick values)
    yticks = (y_zero_tick:y_tick_loc:n, Int32.(round.(0:y_tick_interval:y_max)))
    ;kwargs...)

    return p
end

function plot_PSRH(group_selected,
    idx_files,
    save_name;
    #ECM_dir = "data/4000x4000_combined/subregion_ECM/",
    PSRH_dir = "data/4000x4000_combined/PSRH/",
    grid_layout = nothing,
    size = nothing,
    bottom_margin = 0mm)
    plot_array = []
    n_group = length(group_selected)
    n_ROI = length(group_selected[1])
    for i=1:n_group
        R = group_selected[i]
        for idx in R
            f = idx_files[idx]

            #p_ECM = Images.load(ECM_dir * f * ".tif" )
            p_PSRH = Images.load(PSRH_dir * f * ".tif")
            #push!(plot_array, plot(p_ECM, ticks = [], frame = :box))
            push!(plot_array, plot(p_PSRH, ticks = [], frame = :box))
        end
    end

    if grid_layout == nothing
        grid_layout = grid(n_group, n_ROI)
    end

    if size == nothing
        size = (250 * n_ROI, 250 * n_group)
    end
    p = plot(plot_array..., layout = grid_layout, size = size)
    savefig(save_name)
end

function plot_dim_red2(y, groups; 
    dim_red = "UMAP", 
    xaxis = "UMAP-1", 
    yaxis = "UMAP-2", 
    kwargs...)
    """
    groups: Dictionary of form (i => [indices])
    """

    ### specify colors ###
    c = Dict(
    0 => :grey,
    1 => "#fd5184", # pink
    2=> "#F28522", # orange
    3 => "#ffb602", # yellow
    4 => "#AEF359", # lime green
    5 => "#49a849", # green
    6 => "#3ec0c9", # blue / teal 
    7 => "#265BF5", # dark blue 
    8 => "#8B008B", # purple 
    9 => "#215467", # ocean,
    10 => :black
    )

    ### marker shapes ###
    markershapes = Dict(1 => :rect,
            2 => :utriangle,
            3 => :star,
            4 => :pentagon,
            5 => :diamond,
            6 => :dtriangle,
            7 => :star8,
            8 => :octagon,
            9 => :star4,
            10 => :pentagon
            )
    ### label
    if dim_red == "UMAP"
        label_prefix = "U"
    else
        label_prefix = "R"
    end

    # other figure parameters
    markersize = 3
    legendfontsize = 5
    markerstrokewidth = 1

    n = size(y, 2)
    n_groups = length(groups)

    annotated = vcat([val for (key, val) in groups]...)
    nonannotated = [i for i = 1:size(y, 2) if i ∉ annotated]
    p = scatter(y[1,nonannotated], y[2,nonannotated];
                markercolor = "lightgray",
                alpha = 0.6,
                markersize = markersize, 
                markerstrokewidth = markerstrokewidth,
                label = "", 
                xaxis = xaxis,
                yaxis = yaxis,
                ticks = [],
                guidefontsize = 7,
                framestyle = :box,
                size = (200, 150),
                background_color=:transparent, foreground_color=:black,
                kwargs...
    )


    for k = 1:n_groups
        v = groups[k]
        scatter!(y[1,v], y[2,v], 
                markersize = markersize,
                markerstrokewidth = markerstrokewidth,
                markershape = markershapes[k],
                c = c[k],
                labels = label_prefix * string(k),
                legendfontsize = legendfontsize
                )     
    end
    return p

end

function load_PI_dict(dir)
    PI_dict = Dict()
    dir_files = [item for item in walkdir(dir)][1][3]
    files = [item[1:end-4] for item in dir_files]
    for f in files
        PI_dict[f] = Array(CSV.read(dir * f * ".csv", header = false))
    end
    return PI_dict
end

end