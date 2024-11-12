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

def plot_3D_interactive_plotly(adata, obs_column_name =None, gene_name=None, values = None,
                              pixelsize_xy = 1, pixelsize_z = 1, units = 'um', markersize = 1, opacity = 0.8,
                              obsm_positions = 'spatial_affine_postreg', cmin = None, cmax = None, colormap = 'Reds'):
    #adata: AnnData object; obs_column_name: name of the column in adata.obs to visualise; values: visualise only this values from obs_column_name; gene_name: name of gene expression to visualise 
    #pixelsize_xy: size of the pixel in xy in a chosen units; pixelsize_z - size of the pixel in z in a chosen units; units: chosen units  only for display; markersize: size of one visium spot; opacity: opacity of the spot
    #obsm_positions: name of the attribute in obsm to use as spot positions, cmin, cmax: min, max intensity value for gene expression for colormap; colormap: name of colormap to use
        
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()
    if obs_column_name and gene_name:
        adata_filtered = filter_anndata(adata, obs_column_name, values)
        x,y,z = create_3D_spot_pos(adata_filtered, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        #prepare gene expression for this particular gene
        id_gene = adata_filtered.var['SYMBOL']==gene_name
        idx = np.where(id_gene==True)[0][0]
        gene_expr = adata_filtered.X[:,idx].toarray()[:,0]
        trace = go.Scatter3d(x=x,  y=y, z=z, mode='markers', name="",
                             marker={'size': markersize, 'opacity': opacity, 'color': gene_expr, 
                                     'colorscale': colormap, 'showscale': True, 'cmin': cmin, 'cmax': cmax})
        
    elif obs_column_name and not gene_name:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        #prepare color list
        column = adata.obs[obs_column_name]
        color_list, roi_list, colormaps = prepare_color_list(column, values)

        # Configure the trace.
        trace = go.Scatter3d(x=x,  y=y, z=z, mode='markers', marker={'size': markersize, 'opacity': opacity, 'color': color_list},
        name="")

    elif gene_name and not obs_column_name:
        x,y,z = create_3D_spot_pos(adata, obsm_positions)
        x*=pixelsize_xy; y*=pixelsize_xy; z*=pixelsize_z
        #prepare gene expression for this particular gene
        id_gene = adata.var['SYMBOL']==gene_name
        idx = np.where(id_gene==True)[0][0]
        gene_expr = adata.X[:,idx].toarray()[:,0]
        if not cmin: cmin = np.min(gene_expr)
        if not cmax: cmax = np.max(gene_expr)
        # Configure the trace.
        trace = go.Scatter3d(x=x,  y=y, z=z, mode='markers', name="",
                             marker={'size': markersize, 'opacity': opacity, 'color': gene_expr, 
                                     'colorscale': colormap, 'showscale': True, 'cmin': cmin, 'cmax': cmax})
    else:
        raise ValueError("Please specify gene_name or obs_column_name!")

    

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
    if obs_column_name and not gene_name:
        add_legend(plot_figure, roi_list, colormaps, obs_column_name)
    '''
    plot_figure.update_layout(scene = dict(
                      xaxis=dict(title=dict(text='X, ' + str(units))),
                      yaxis=dict(title=dict(text='Y, ' + str(units))),
                      zaxis=dict(title=dict(text='Z, ' + str(units)),),))
    '''
    # Render the plot.
    plotly.offline.iplot(plot_figure)
