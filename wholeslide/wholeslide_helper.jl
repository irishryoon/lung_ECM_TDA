module wholeslide_helper
include("../src/ECM_TDA.jl")
using .ECM_TDA
using Plots
using Images
using CSV
using DataFrames
using JLD2
using FileIO
using Distributions
using Statistics
using Measures
using TiffImages
using Plots
using Images
using StatsBase

export plot_ECM_collection,
        plot_ECM_collection2, 
        plot_wholeslide,
        plot_wholeslide_points_colored_by_clusters,
        get_row_col_lim,
        stitch_2000_tiles_to_4000,
        get_ECM_mean_pixels,
        sample_points_from_ECM_directory,
        assign_cluster,
        load_tile_data,
        get_concatenated_features,
        get_closest_clusters,
        plot_wholeslide_points_colored_by_clusters2,
        combine_dim01_PIs,
        find_closest_cluster_for_wholeslide_tiles

function get_row_col_lim(dir_2000)
    files =  [item for item in walkdir(dir_2000)][1][3]
    files = [x for x in files if (x[end-3:end] == ".tif") & (x[1] != '.')]

    rows = [parse(Int32, split(f, "_")[3]) for f in files]
    cols = [parse(Int32, split(f,"_")[4]) for f in files]

    max_row = maximum(rows)
    max_col = maximum(cols)

    if iseven(max_row)
        row_lim = max_row - 2
    else
        row_lim = max_row
    end

    if iseven(max_col)
        col_lim = max_col - 2
    else
        col_lim = max_col
    end
    return row_lim, col_lim

end

function stitch_2000_tiles_to_4000(dir_2000, dir_4000, row_lim, col_lim)
    # create paths
    ispath(dir_4000) || mkpath(dir_4000)

    for i =0:2:row_lim
        for j = 0:2:col_lim
            
            input1 = dir_2000 * "image_tile_" * lpad(i, 5, "0") * "_" * lpad(j, 5, "0") * "_psr.tif"
            input2 = dir_2000 * "image_tile_" * lpad(i, 5, "0") *  "_" * lpad(j+1, 5, "0") * "_psr.tif"
            input3 = dir_2000 * "image_tile_" * lpad(i+1, 5, "0") * "_" * lpad(j, 5, "0") * "_psr.tif"
            input4 = dir_2000 * "image_tile_" * lpad(i+1, 5, "0") * "_" * lpad(j+1, 5, "0") * "_psr.tif"
            
            new_i = lpad(Int32(i/2), 5, "0")
            new_j = lpad(Int32(j/2), 5, "0")
    
            outputfile = dir_4000 * new_i * "_" * new_j * "_psr.tif"
            mycommand = `magick montage -tile 2x2 -geometry +0+0 $input1 $input2 $input3 $input4 $outputfile`
            run(mycommand)
        end
    end
end

function plot_ECM_collection(regions, idx_ROI, ECM_dir)
    gr()
    plot_array = []
    n = length(regions)
    for i=1:n
        R = regions[i]
        for idx in R
            f = idx_ROI[idx]
            p = Images.load(ECM_dir * f * ".tif" )
            push!(plot_array, plot(p, ticks = [], frame = :box))
        end
    end

    return plot_array
end

function plot_wholeslide(idx_clusters, ROI_idx; CSV_directory = "", filename = "", markersize = markersize)

    p = plot()

    # get all CSV files 
    csv_files = [item for item in walkdir(CSV_directory)][1][3]
    for f in csv_files
        prefix = split(f, ".")[1]
        row, col, _ = split(f, "_")
        row = parse(Int32, row)
        col = parse(Int32, col)
        
        row_adjust = row * 4000
        col_adjust = col * 4000
        
        # load points
        df = CSV.read(CSV_directory * f)

        # plot
        scatter!(p, df[:,:x] .+ col_adjust, df[:,:y] .+ row_adjust, 
                yflip = :true,
                label = "", markersize = markersize, markerstrokewidth = 0,
                c = :grey
        )
    end

    plot(p, aspect_ratio = :equal, size = (5000, 5000))
    savefig(filename)
end

function plot_wholeslide_points_colored_by_clusters(idx_clusters, ROI_idx; CSV_directory = "", filename = "", markersize = markersize)

    # specify colors
    colors = Dict(
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

    c_none = :grey
    
    p = plot()

    # get all CSV files 
    csv_files = [item for item in walkdir(CSV_directory)][1][3]
    for f in csv_files
        prefix = split(f, ".")[1]
        row, col, _ = split(f, "_")
        row = parse(Int32, row)
        col = parse(Int32, col)
        
        row_adjust = row * 4000
        col_adjust = col * 4000
        
        # load points
        df = CSV.read(CSV_directory * f)

        # find cluster
        if (prefix in keys(ROI_idx)) && (ROI_idx[prefix] in keys(idx_clusters))
            cluster = idx_clusters[ROI_idx[prefix]]
            c = colors[cluster]
        else 
            c = c_none
        end
        
    
        scatter!(p, df[:,:x] .+ col_adjust, df[:,:y] .+ row_adjust, 
                yflip = :true,
                label = "", markersize = markersize, markerstrokewidth = 0,
                c = c
        )
    end

    plot(p, aspect_ratio = :equal, size = (5000, 5000))
    savefig(filename)
end

function get_ECM_mean_pixels(directory)
    files =  [item for item in walkdir(directory)][1][3]
    files = [f for f in files if f[1] != '.']
    ECM_mean_pixels = []
    for filename in files
        img = Array(Images.load(directory * filename))
        push!(ECM_mean_pixels, mean(Float64.(img)))
    end
    return ECM_mean_pixels
end

function sample_points_from_ECM_directory(ECM_directory, min_sample, max_sample, low, high, c_ECM;
                                         invert = true,
                                         sample_plot_directory = "",
                                         sample_CSV_directory = "")
    # create directories for sample plot and CSV if it doesn't exist
    if sample_plot_directory != "" 
        ispath(sample_plot_directory) || mkpath(sample_plot_directory)
    end
    if sample_CSV_directory != ""
        ispath(sample_CSV_directory) || mkpath(sample_CSV_directory)
    end


    files = [item for item in walkdir(ECM_directory)][1][3]
    for filename in files
        if filename[1] != '.'
            img = Array(Images.load(ECM_directory * filename))
            figure_file = split(filename,".")[1] * ".pdf"

            # compute (inverted) mean pixel value of image
            #img_mean_inv = 1- mean(Float64.(img))
            img_mean_inv = mean(Float64.(img))

            # compute number of points to sample
            n_sample = Int64(round(ECM_TDA.n_ECM_samples(img_mean_inv, low, high, min_sample, max_sample)))

            # sample points
            resampled, points, sampled, island_idx = sample_ECM_points(img, 
                                                                        vicinity = 100, 
                                                                        n_points = 5, 
                                                                        n_samples = n_sample,
                                                                        invert = invert)

            # plot the results 
            p = scatter(resampled[:,1], resampled[:,2], yflip = :true, c = c_ECM, label = "", frame = :box, ticks = [], 
                        aspect_ratio = :equal,
                        xlims = (0, 4000),
                        ylims = (0, 4000),
                        size = (500, 500))
            savefig(sample_plot_directory * figure_file)

            # save sampled points to CSV
            csv_file = split(filename, ".")[1] * ".csv"
            df = DataFrame(resampled, [:x, :y])
            CSV.write(sample_CSV_directory * csv_file, df)
        end
    end
end


function plot_ECM_collection2(regions, idx_ROI, ECM_dir)
    gr()
    plot_array = []
    n = length(regions)
    for i=-1:n-2
        R = regions[i]
        for idx in R
            if idx != -1
                f = idx_ROI[idx]
                p = Images.load(ECM_dir * f * ".tif" )
                push!(plot_array, plot(p, ticks = [], frame = :box))
            else
                push!(plot_array, plot(ticks = [], frame = :box))
            end
        end
    end

    return plot_array
end


function assign_cluster(closest_clusters, k)
    assigned_clusters = Dict()
    for i = 1:size(closest_clusters,1)
        m = mode(closest_clusters[i,1:k])
        if counts(Int64.(closest_clusters[i,1:k]))[1] == 1
            assigned_clusters[i] = 0
        else
            assigned_clusters[i] = m
        end
    end
    return assigned_clusters
end

function load_tile_data(LTX; normal = false)
    if normal == false
        # load PD, PI from tiles images (dataset 3)
        PD = load("data_TDA/LTX" * string(LTX) * "/PD.jld2")
        PD0 = PD["PD0"]
        PD1 = PD["PD1"]
    else
        PD = load("data_TDA/normal_LTX" * string(LTX) * "/PD.jld2")
        PD0 = PD["PD0"]
        PD1 = PD["PD1"]
    end

    # recompute coarser PI
    PH0_dict = Dict(k => ECM_TDA.array_to_ripsererPD(v) for (k,v) in PD0 if v != nothing);
    PH1_dict = Dict(k => ECM_TDA.array_to_ripsererPD(v) for (k,v) in PD1 if v != nothing);

    PI0 = PersistenceImage([PH0_dict[k] for k in keys(PH0_dict)], sigma=50, size = 20)
    PI1 = PersistenceImage([PH1_dict[k] for k in keys(PH1_dict)], sigma=50, size = 20)


    ECM_PI0 = Dict()
    for i in keys(PH0_dict)
        ECM_PI0[i] = PI0(PH0_dict[i])
    end

    ECM_PI1 = Dict()
    for i in keys(PH1_dict)
        ECM_PI1[i] = PI1(PH1_dict[i])
    end
    return ECM_PI0, ECM_PI1

end

function get_concatenated_features(LTX)
    # if features and idx_ROI already exist, load
    if isfile("data_TDA/LTX" * string(LTX) * "/idx_ROI_PI01.jld2")
        idx_ROI = load("data_TDA/LTX" * string(LTX) * "/idx_ROI_PI01.jld2")["idx_files"]
        ROI_idx = Dict(v => k for (k,v) in idx_ROI)
        features = load("data_TDA/LTX" * string(LTX) * "/PI01_features.jld2")["features"]

        return features, idx_ROI, ROI_idx

    # else: concatenate features and save
    else
        # concatenate dimension 0 and 1 features
        features = Dict()
        for f in keys(PI0)
             # check that f is a key in all dictionaries
              if f in keys(PI1)
                  combined = vcat(PI0[f], vec(PI1[f]))
                  features[f] = combined
              end
        end

        ROIs = collect(keys(features))
        idx_ROI = Dict(i => roi for (i, roi) in enumerate(ROIs));

        # # get features array
        n = length(idx_ROI)
        features_array = hcat([features[idx_ROI[i]] for i = 1:n]...)

        save("data_TDA/LTX" * string(LTX) * "/idx_ROI_PI01.jld2", "idx_files", idx_ROI)
        save("data_TDA/LTX" * string(LTX) * "/PI01_features.jld2", "features", features_array)
        ROI_idx = Dict(v => k for (k,v) in idx_ROI)

        return features_array, idx_ROI, ROI_idx

    end
end


"""get_closest_clusters()

Given a matrix whose rows: indices of tile data, columns: indices of original data. \n
return a matrix whose entriy at (i,j) is the cluster of j"""
function get_closest_clusters(closest_indices, idx_clusters_original)
    n_rows, n_cols = size(closest_indices)
    closest_clusters = zeros((n_rows, n_cols))
    for i = 1:n_rows
        for j = 1:n_cols
            k = Int32(closest_indices[i,j])
            if k in keys(idx_clusters_original)
                closest_clusters[i,j] = idx_clusters_original[k]
            end
        end
    end
    return closest_clusters
end

function plot_wholeslide_points_colored_by_clusters2(idx_clusters, ROI_idx; CSV_directory = "", filename = "", markersize = 2, size = (5000, 5000))

    # specify colors
    colors = Dict(
        -1 => "#A9A9A9",
        0 => "#780000", 
        1=> "#cb334c", 
        2 => "#f89981",
        3 => "#ffbd00",
        4 => "#02c39a",
        5 => "#429bb4",
        6 => "#7851A9",
        7 => "#32174D"
        )
    c_none = :grey
    
    p = plot()

    # get all CSV files 
    csv_files = [item for item in walkdir(CSV_directory)][1][3]
    for f in csv_files
        prefix = split(f, ".")[1]
        row, col, _ = split(f, "_")
        row = parse(Int32, row)
        col = parse(Int32, col)
        
        row_adjust = row * 4000
        col_adjust = col * 4000
        
        # load points
        df = CSV.read(CSV_directory * f)

        # find cluster
        if (prefix in keys(ROI_idx)) && (ROI_idx[prefix] in keys(idx_clusters))
            cluster = idx_clusters[ROI_idx[prefix]]
            c = colors[cluster]
        else 
            c = c_none
        end
        
    
        scatter!(p, df[:,:x] .+ col_adjust, df[:,:y] .+ row_adjust, 
                yflip = :true,
                label = "", markersize = markersize, markerstrokewidth = 0,
                c = c,
                ticks = [],
                xaxis = false,
                yaxis = false
        )
    end

    plot(p, aspect_ratio = :equal, size = size)
    savefig(filename)
end


function combine_dim01_PIs(PI0, PI1)
    features_dict = Dict()
    for f in keys(PI0)
        # check that f is a key in all dictionaries
        if f in keys(PI1)
            combined = vcat(PI0[f], vec(PI1[f]))
            features_dict[f] = combined
        end
    end
    return features_dict
end

function find_closest_cluster_for_wholeslide_tiles(features, avg_features_by_clusters)
    # for each tile (of wholeslide image), find closest cluster via average
    n_ROIs = size(features, 2)
    assigned_clusters = Dict()
    n_clusters = length(avg_features_by_clusters)
    for i = 1:n_ROIs
        # compute distance to average features of original clusters_original
        distances = []
        for j = -1:n_clusters - 2
            d = euclidean(features[:,i], avg_features_by_clusters[j])
            push!(distances, d)
        end
        # find cluster with min distance
        min_cluster = argmin(distances) - 2
        assigned_clusters[i] = min_cluster
    end
    return assigned_clusters
end

end