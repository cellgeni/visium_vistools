import numpy as np
import pandas as pd
import anndata
import cv2 as cv
import plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

def create_3D_spot_pos(adata, obsm_positions):
    list_of_section_names = list(adata.uns['spatial'].keys())
    x = np.array([]); y = np.array([]); z = np.array([]);
    Zpos = 0;
    for section_name in list_of_section_names:
        idx = adata.obs.index.str.contains(section_name)
        spot_pos_section = adata.obsm[obsm_positions][idx]
        x = np.append(x, spot_pos_section[:,0])
        y = np.append(y, spot_pos_section[:,1])
        zz = np.ones((spot_pos_section.shape[0]))*Zpos
        z = np.append(z, zz)
        Zpos+=1
    return x, y, z

 
    
def prepare_color_list(column, values, grey_color = '#eeeeee'):
    colormap_discrete = ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7','#000000']
    
    if not values:
        values = np.unique(column)
    color_list = [grey_color]*column.shape[0]  
    i=0
    for roi in values:
        idx = column==roi
        idx = np.where(idx.values==True)[0]
        #print(idx)
        for idd in idx:
            #print(idd)
            color_list[idd] = colormap_discrete[i%len(colormap_discrete)]
        i+=1
    return color_list, values, colormap_discrete

def add_legend(fig, roi_list, colormaps, title):
    
    if not title:
        title = ' '
    for i, roi in zip(range(len(roi_list)), roi_list):
        color = colormaps[i%len(colormaps)]
        fig.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[2, 1, 3],
        legendgrouptitle_text=title,
        legendgroup="group",  # this can be any string, not just "group"
        name=roi,
        mode="markers",
        marker=dict(color=color, size=10)))

def get_color_opacity_arrays(values, cmin, cmax, color_final = [213, 94, 0]):
    if not cmin: cmin = np.min(values)
    if not cmax: cmax = np.max(values)
    opacity_array = prepare_opacity_array(values, cmin, cmax)#.astype(str)
    color_list_R, color_list_G, color_list_B = get_color_val(values, [0,0,0], color_final, cmin, cmax)
    return opacity_array, color_list_R, color_list_G, color_list_B, cmin, cmax

        
def prepare_opacity_array(values_array, cmin=None, cmax=None):
    return (values_array-cmin)/(cmax-cmin)

def add_legend_continuous(fig, colors_list, cmin_list, cmax_list, gene_list):
    
    for i in range(len(gene_list)):
        col = colors_list[i]; cmin = cmin_list[i]; cmax = cmax_list[i]
        #col_s = [f'rgb({int(col[0])}, {int(col[1])}, {int(col[2])})]
        col_s = 'rgb(' + str(int(col[0])) + ', ' + str(int(col[1])) + ', ' + str(int(col[2])) + ')'
        fig.add_trace(go.Scatter(
        x=[1],
        y=[2],
        #legendgrouptitle_text=title,
        #legendgroup="group",  # this can be any string, not just "group"
        name=gene_list[i] + ' (' + str(cmin) + '---' + str(cmax) + ')',
        mode="markers",
        marker=dict(color=col_s, size=10)))


def prepare_trace_continuous(adata, gene_names, x, y, z, cmin, cmax, markersize):
    #hardcoded colors taken from c2l
    final_color_list = [[90, 20, 165], [213, 94, 0], [0, 158, 115], [86, 180, 233], [240, 228, 66], [200, 200, 200], [50, 50, 50]]
    #cycle is meant to mix colors and opacity in case of several genes(sets) to be displayed
    cmin_list = []; cmax_list = []
    for i in range(np.min([len(final_color_list), len(gene_names)])):
        id_gene = adata.var['SYMBOL']==gene_names[i]
        idx = np.where(id_gene==True)[0][0]
        gene_expr = adata.X[:,idx].toarray()[:,0]
        opacity_array, color_list_R, color_list_G, color_list_B, cmin1, cmax1 = get_color_opacity_arrays(gene_expr, cmin, cmax, color_final = final_color_list[i])
        cmin_list.append(cmin1); cmax_list.append(cmax1); 
        if i==0:
            opacity_aggr, color_R_aggr, color_G_aggr, color_B_aggr = opacity_array, color_list_R, color_list_G, color_list_B
        else:
            opacity_aggr+=opacity_array; 
            color_R_aggr+=color_list_R; color_G_aggr+=color_list_G; color_B_aggr+=color_list_B;
        
    #find average:
    opacity_aggr/=(i+1); color_R_aggr/=(i+1); color_G_aggr/=(i+1); color_B_aggr/=(i+1); 
    opacity_aggr = opacity_aggr.astype('str')

    #construct a dictionary for color and opacity
    color_opacity_dict = [ f'rgba({int(color_R_aggr[i])}, {int(color_G_aggr[i])}, {int(color_B_aggr[i])}, {opacity_aggr[i]})' for i in range(opacity_array.shape[0])]    
    
    trace = go.Scatter3d(x=x,  y=y, z=z, mode='markers', name="",
                             marker={'size': markersize, 'color': color_opacity_dict})
    return trace, final_color_list[:i+1], cmin_list, cmax_list        
        

def filter_anndata(adata, column_name, *filter_values):
    if column_name not in adata.obs:
        raise ValueError(f"Column '{column_name}' does not exist in AnnData object.")
    if filter_values:
        filter_values = filter_values[0]
        mask = adata.obs[column_name].isin(filter_values)
        filtered_adata = adata[mask].copy()
    else:
        filtered_adata = adata
    
    return filtered_adata

def get_color_val(values, color_min, color_max, cmin, cmax):
    values_col = (np.clip(values, cmin, cmax)-cmin)/(cmax-cmin)
    #print(values_col)
    color_list_R = (color_max[0]*values_col + color_min[0]*(1-values_col))
    color_list_G = (color_max[1]*values_col + color_min[1]*(1-values_col))
    color_list_B = (color_max[2]*values_col + color_min[2]*(1-values_col))
    return color_list_R, color_list_G, color_list_B

def set_bgcolor(bg_color = "rgb(20, 20, 20)",
                grid_color="rgb(200, 200, 200)", 
                zeroline=False):
    return dict(showbackground=True,
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                zeroline=zeroline)


def plot_3D_interactive_plotly(adata, obs_column_name =None, gene_names=None, values = None,
                              pixelsize_xy = 1, pixelsize_z = 1, units = 'um', markersize = 1, opacity = 0.8,
                              obsm_positions = 'spatial_affine_postreg', cmin = None, cmax = None, colormap = 'Reds',
                              background_black = False, save_html = None):
    #adata: AnnData object; obs_column_name: name of the column in adata.obs to visualise; values: visualise only this values from obs_column_name; gene_names: list of names of genes expression to visualise (i would not go for more than 5 genes due to the color mixing) 
    #pixelsize_xy: size of the pixel in xy in a chosen units; pixelsize_z - size of the pixel in z in a chosen units; units: chosen units  only for display; markersize: size of one visium spot; opacity: opacity of the spot
    #obsm_positions: name of the attribute in obsm to use as spot positions, cmin, cmax: min, max intensity value for gene expression for colormap; colormap: name of colormap to use
        
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()
    if obs_column_name and gene_names:
        
        adata_filtered = filter_anndata(adata, obs_column_name, values)
        x,y,z = create_3D_spot_pos(adata_filtered, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        trace, colors_list_cont, cmin_list, cmax_list = prepare_trace_continuous(adata_filtered, gene_names, x, y, z, cmin, cmax, markersize)
        
    elif obs_column_name and not gene_names:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        #prepare color list
        column = adata.obs[obs_column_name]
        color_list, roi_list, colormaps = prepare_color_list(column, values)

        # Configure the trace.
        trace = go.Scatter3d(x=x,  y=y, z=z, mode='markers', marker={'size': markersize, 'opacity': opacity, 'color': color_list},
        name="")

    elif gene_names and not obs_column_name:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        trace, colors_list_cont, cmin_list, cmax_list = prepare_trace_continuous(adata, gene_names, x, y, z, cmin, cmax, markersize)

    else:
        raise ValueError("Please specify gene_names or obs_column_name!")

    

    data = [trace]
    layout = go.Layout({'xaxis': {
    'range': [0.2, 1],
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False},
    'yaxis': {
    'range': [0.2, 1],
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False}},
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    plot_figure = go.Figure(data=data, layout=layout)
    

    if obs_column_name and not gene_names:
        add_legend(plot_figure, roi_list, colormaps, obs_column_name)
    else:
        add_legend_continuous(plot_figure, colors_list_cont, cmin_list, cmax_list, gene_names)
      
    if background_black:
        plot_figure.update_layout(paper_bgcolor='rgba(0,0,0,255)', font=dict(color= 'white'))
        plot_figure.update_scenes(xaxis=set_bgcolor(), 
                      yaxis=set_bgcolor(), 
                      zaxis=set_bgcolor())
    
    # Render the plot.
    plotly.offline.iplot(plot_figure)
    if save_html:
        plot_figure.write_html(save_html)
