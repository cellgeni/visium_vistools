import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import anndata
from vis3D_anndata_tools import create_3D_spot_pos, filter_anndata, prepare_color_list, get_color_val, prepare_color_list


#functions


def make_plot_continuous(mode, ranges, pos, color_array, opacity_array, markersize, units, list_of_genes, color_list, cmin_list, cmax_list):
    axis_1 = mode[0]; axis_2 = mode[1]
    for xi in ["x", "y","z"]:
        if xi not in mode:
            axis_3 = xi
    #now we need to chose only the spots which are in perpendicular axis
    idx = (pos[axis_3]>ranges[axis_3][0]) & (pos[axis_3]<ranges[axis_3][1])
    xx = pos[axis_1][idx]; yy = pos[axis_2][idx]; zz = pos[axis_3][idx];
    color_array_c = color_array[idx]; opacity_array_c = opacity_array[idx]; 
    #here I match size of the figure to its range in axis_1 and axis_2
    range_arr = [ranges[axis_1][1]-ranges[axis_1][0], ranges[axis_2][1]-ranges[axis_2][0]]
    max_pos = range_arr.index(max(range_arr)); max_range = np.max(range_arr)
    min_pos = range_arr.index(min(range_arr)); min_range = np.min(range_arr)
    figure_size = [0,0]; figure_size[max_pos] = 20; figure_size[min_pos] = 20*min_range/max_range; 
    fig, ax = plt.subplots(1,1, figsize = (int(figure_size[0]),int(figure_size[1])))
    
    plt.scatter(xx, yy, c = color_array_c, alpha = opacity_array_c, s = markersize)
    plt.axis('equal')
    plt.xlim(ranges[axis_1]); plt.ylim(ranges[axis_2])
    plt.xlabel(str(axis_1) + ', ' + units, fontsize = 20)
    plt.ylabel(str(axis_2) + ', ' + units, fontsize = 20)
    title_text = axis_3 + ' range: ' + ' (' + str(ranges[axis_3][0]) + '__' + str(ranges[axis_3][1]) + ') ' + units
    plt.title(title_text, fontsize = 20)
    make_legend(ax, list_of_genes, cmin_list, cmax_list, color_list)

def make_plot_categorical(mode, ranges, pos, color_array, roi_list, opacity_array, markersize, units, colormaps):
    axis_1 = mode[0]; axis_2 = mode[1]
    for xi in ["x", "y","z"]:
        if xi not in mode:
            axis_3 = xi
    #now we need to chose only the spots which are in perpendicular axis
    idx = (pos[axis_3]>ranges[axis_3][0]) & (pos[axis_3]<ranges[axis_3][1])
    xx = pos[axis_1][idx]; yy = pos[axis_2][idx]; zz = pos[axis_3][idx];
    color_array = np.array(color_array)

    color_array_c = color_array[idx];
    opacity_array_c = opacity_array[idx]; 
    
    #here I match size of the figure to its range in axis_1 and axis_2
    range_arr = [ranges[axis_1][1]-ranges[axis_1][0], ranges[axis_2][1]-ranges[axis_2][0]]
    max_pos = range_arr.index(max(range_arr)); max_range = np.max(range_arr)
    min_pos = range_arr.index(min(range_arr)); min_range = np.min(range_arr)
    figure_size = [0,0]; figure_size[max_pos] = 20; figure_size[min_pos] = 20*min_range/max_range; 
    fig, ax = plt.subplots(1,1, figsize = (int(figure_size[0]),int(figure_size[1])))
    
    plt.scatter(xx, yy, c = color_array_c, alpha = opacity_array_c, s = markersize)
    plt.axis('equal')
    plt.xlim(ranges[axis_1]); plt.ylim(ranges[axis_2])
    plt.xlabel(str(axis_1) + ', ' + units, fontsize = 20)
    plt.ylabel(str(axis_2) + ', ' + units, fontsize = 20)
    title_text = axis_3 + ' range: ' + ' (' + str(ranges[axis_3][0]) + '__' + str(ranges[axis_3][1]) + ') ' + units
    plt.title(title_text, fontsize = 20)
    make_legend_cat(ax, roi_list, colormaps)
    
def get_opacity_array(mode, ranges, pos):
    for xi in ["x", "y","z"]:
        if xi not in mode:
            perp_axis = xi    
    opacity_array = prep_opacity(pos[xi], ranges[xi][0], ranges[xi][1])
    return opacity_array

def prep_xyz_range(x_range, y_range, z_range, x, y, z):
    if not x_range: x_range = [np.min(x), np.max(x)]
    if not y_range: y_range = [np.min(y), np.max(y)]
    if not z_range: z_range = [np.min(z), np.max(z)]
    return x_range, y_range, z_range

def make_legend(ax, list_of_genes, cmin_list, cmax_list, colors):
    legend_elements = []
    for i in range(len(list_of_genes)):
        string_show = list_of_genes[i] + ' (' + str(cmin_list[i]) + '--' + str(cmax_list[i]) + ')'
        legend_elements.append(Line2D([0], [0], marker='o', color = 'w',  markerfacecolor = np.array(colors[i])/255, label = string_show, markersize = 10))
        ax.legend(handles=legend_elements)

        
def make_legend_cat(ax, list_of_rois, colors):
    legend_elements = []
    for i in range(len(list_of_rois)):
        legend_elements.append(Line2D([0], [0], marker='o', color = 'w',  markerfacecolor = colors[i], label = list_of_rois[i], markersize = 10))
        ax.legend(handles=legend_elements)

        
def plot_one_continous_column(mode, column, name_column, ranges, pos, markersize, units, cmin, cmax, colormap = 'Reds'):
    axis_1 = mode[0]; axis_2 = mode[1]
    for xi in ["x", "y","z"]:
        if xi not in mode:
            axis_3 = xi
    #now we need to chose only the spots which are in perpendicular axis
    idx = (pos[axis_3]>ranges[axis_3][0]) & (pos[axis_3]<ranges[axis_3][1])
    xx = pos[axis_1][idx]; yy = pos[axis_2][idx]; zz = pos[axis_3][idx];
    opacity_array = get_opacity_array(mode, ranges, pos);
    opacity_array_c = opacity_array[idx];
    #here I match size of the figure to its range in axis_1 and axis_2
    range_arr = [ranges[axis_1][1]-ranges[axis_1][0], ranges[axis_2][1]-ranges[axis_2][0]]
    max_pos = range_arr.index(max(range_arr)); max_range = np.max(range_arr)
    min_pos = range_arr.index(min(range_arr)); min_range = np.min(range_arr)
    figure_size = [0,0]; figure_size[max_pos] = 20; figure_size[min_pos] = 20*min_range/max_range; 
    fig, ax = plt.subplots(1,1, figsize = (int(figure_size[0]),int(figure_size[1])))
    if not cmin: cmin = [np.min(column[idx])]
    if not cmax: cmax = [np.max(column[idx])]
    plot = plt.scatter(xx, yy, c = column[idx], alpha = opacity_array_c, s = markersize, cmap = colormap, vmin = cmin[0], vmax = cmax[0])
    plt.axis('equal')
    plt.xlim(ranges[axis_1]); plt.ylim(ranges[axis_2])
    plt.xlabel(str(axis_1) + ', ' + units, fontsize = 20)
    plt.ylabel(str(axis_2) + ', ' + units, fontsize = 20)
    title_text = axis_3 + ' range: ' + ' (' + str(ranges[axis_3][0]) + '__' + str(ranges[axis_3][1]) + ') ' + units
    plt.title(title_text, fontsize = 20)
    norm = mpl.colors.Normalize(vmin=cmin[0], vmax=cmax[0])
    fig.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label = name_column)
    #make_legend(ax, name_column, )

def plot_all_genes(adata, gene_names, x, y, z, x_range, y_range, z_range, cmin_all, cmax_all, mode, markersize, units):
    #mode can be either "xy", "xz", "zy", "zx", "yx", "yz"
    #hardcoded colors taken from c2l
    final_color_list = [[90, 20, 165], [213, 94, 0], [0, 158, 115], [86, 180, 233], [240, 228, 66], [200, 200, 200], [50, 50, 50]]
    #cycle is meant to mix colors and opacity in case of several genes(sets) to be displayed
    cmin_list = []; cmax_list = []
    ranges = {"x":x_range, "y":y_range, "z":z_range}; pos = {"x":x, "y":y, "z":z}
    opacity_array = get_opacity_array(mode[7:], ranges, pos)
    for i in range(np.min([len(final_color_list), len(gene_names)])):
        id_gene = adata.var['SYMBOL']==gene_names[i]
        idx = np.where(id_gene==True)[0][0]
        gene_expr = adata.X[:,idx].toarray()[:,0]
        if cmin_all:
            cmin = cmin_all[i]
        else:
            cmin = cmin_all
        if cmax_all:
            cmax = cmax_all[i]
        else:
            cmax = cmax_all
        color_list_R, color_list_G, color_list_B, cmin, cmax = get_color_arrays(gene_expr, cmin, cmax, final_color_list[i])
        cmin_list.append(cmin); cmax_list.append(cmax)
        if i==0:
            color_R_aggr, color_G_aggr, color_B_aggr = color_list_R, color_list_G, color_list_B
        else:
            color_R_aggr+=color_list_R; color_G_aggr+=color_list_G; color_B_aggr+=color_list_B;
    #find average:
    color_R_aggr/=(i+1); color_G_aggr/=(i+1); color_B_aggr/=(i+1); 
    color_array = np.zeros((len(color_R_aggr),3)); 
    color_array[:,0] = color_R_aggr/255; color_array[:,1] = color_G_aggr/255; color_array[:,2] = color_B_aggr/255; 
    
    make_plot_continuous(mode[7:], ranges, pos, color_array, opacity_array, markersize, units, gene_names, final_color_list[:i+1], cmin_list, cmax_list)
    
def prep_opacity(values, min_value, max_value, min_opacity = 0.5):
    values_c = np.clip(values, min_value, max_value)
    return (values_c-min_value)/(max_value-min_value)*(1-min_opacity)+min_opacity

def get_color_arrays(values, cmin, cmax, color_final = [213, 94, 0]):
    #firstly we find the axis perpendicular to the viewing plane
    if not cmin: cmin = np.min(values)
    if not cmax: cmax = np.max(values)
    color_list_R, color_list_G, color_list_B = get_color_val(values, [255,255,255], color_final, cmin, cmax)
    return color_list_R, color_list_G, color_list_B, cmin, cmax

def plot_proj(adata, x_range=None, y_range=None, z_range=None, cmin =None, cmax = None, 
              obs_column_name =None, values=None, gene_names=None, mode = 'single_xy', pixelsize_xy = 1, pixelsize_z = 1, 
              obsm_positions = 'spatial_affine_postreg', markersize = 5, units = 'um', colormap = "Reds"):

    if obs_column_name and gene_names:
        adata_filtered = filter_anndata(adata, obs_column_name, values)
        x,y,z = create_3D_spot_pos(adata_filtered, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        x_range, y_range, z_range = prep_xyz_range(x_range, y_range, z_range, x, y, z)
        plot_all_genes(adata_filtered, gene_names, x, y, z, x_range, y_range, z_range, cmin, cmax, mode, markersize, units)
    elif obs_column_name and not gene_names:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        x_range, y_range, z_range = prep_xyz_range(x_range, y_range, z_range, x, y, z)
        ranges = {"x":x_range, "y":y_range, "z":z_range}; pos = {"x":x, "y":y, "z":z}
        column = adata.obs[obs_column_name]
        if column.dtype.name == 'category':
            color_list, roi_list, colormaps = prepare_color_list(column, values)   
            opacity_array = get_opacity_array(mode[7:], ranges, pos)
            make_plot_categorical(mode[7:], ranges, pos, color_list, roi_list, opacity_array, markersize, units, colormaps)
        else:
            plot_one_continous_column(mode[7:], column, obs_column_name, ranges, pos, markersize, units, cmin, cmax, colormap)
    elif gene_names and not obs_column_name:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        x_range, y_range, z_range = prep_xyz_range(x_range, y_range, z_range, x, y, z)
        plot_all_genes(adata, gene_names, x, y, z, x_range, y_range, z_range, cmin, cmax, mode, markersize, units)
    else:
        raise ValueError("Please specify gene_names or obs_column_name!")
    
