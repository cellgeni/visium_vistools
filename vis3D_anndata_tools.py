import numpy as np
import pandas as pd
import anndata
import cv2 as cv
import plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

def _get_section_mask(adata, section_name, section_obs_col="section"):
    if section_obs_col in adata.obs.columns:
        return (adata.obs[section_obs_col].astype(str) == str(section_name)).to_numpy()
    return adata.obs.index.astype(str).str.contains(str(section_name), regex=False).to_numpy()


def create_3D_spot_pos(
    adata,
    obsm_positions,
    use_obsm_z=True,
    section_obs_col="section",
    debug=False,
):
    list_of_section_names = list(adata.uns["spatial"].keys())

    x = np.array([], dtype=float)
    y = np.array([], dtype=float)
    z = np.array([], dtype=float)
    row_order = []

    Zpos = 0
    for section_name in list_of_section_names:
        idx = _get_section_mask(adata, section_name, section_obs_col=section_obs_col)
        selected_idx = np.where(idx)[0]
        n = int(idx.sum())

        if debug:
            print(f"{section_name}: {n} cells")

        if n == 0:
            continue

        spot_pos_section = np.asarray(adata.obsm[obsm_positions][idx], dtype=float)

        if spot_pos_section.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{obsm_positions}'] must have at least 2 columns"
            )

        x = np.append(x, spot_pos_section[:, 0])
        y = np.append(y, spot_pos_section[:, 1])

        if use_obsm_z and spot_pos_section.shape[1] >= 3:
            zz = spot_pos_section[:, 2]
        else:
            zz = np.full(spot_pos_section.shape[0], Zpos, dtype=float)
            Zpos += 1

        z = np.append(z, zz)
        row_order.extend(selected_idx.tolist())

        if debug:
            print(f"  z unique: {np.unique(zz)}")

    return x, y, z, np.array(row_order, dtype=int)


def create_3D_spot_pos_thickness_from_table(
    adata,
    obsm_positions,
    thickness_table,
    thickness_section=5,
    use_obsm_z=False,
    section_obs_col="section",
    debug=False,
):
    list_of_section_names = list(adata.uns["spatial"].keys())

    x = np.array([], dtype=float)
    y = np.array([], dtype=float)
    z = np.array([], dtype=float)
    row_order = []

    for section_name in list_of_section_names:
        idx = _get_section_mask(adata, section_name, section_obs_col=section_obs_col)
        selected_idx = np.where(idx)[0]
        n = int(idx.sum())

        if debug:
            print(f"{section_name}: {n} cells")

        if n == 0:
            continue

        spot_pos_section = np.asarray(adata.obsm[obsm_positions][idx], dtype=float)

        if spot_pos_section.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{obsm_positions}'] must have at least 2 columns"
            )

        x = np.append(x, spot_pos_section[:, 0])
        y = np.append(y, spot_pos_section[:, 1])

        if use_obsm_z and spot_pos_section.shape[1] >= 3:
            zz = spot_pos_section[:, 2]
        else:
            Zpos = getzpos_from_section_name(
                thickness_table,
                section_name,
                thickness_section,
            )
            zz = np.full(spot_pos_section.shape[0], Zpos, dtype=float)

        z = np.append(z, zz)
        row_order.extend(selected_idx.tolist())

        if debug:
            print(f"  z unique: {np.unique(zz)}")

    return x, y, z, np.array(row_order, dtype=int)


def getzpos_from_section_name(thickness_table, section_name, thickness_section = 5):
    section_number = int(section_name[-3:])
    return np.sum(thickness_table['distance_between_sections'][:section_number+1]) + section_number*thickness_section 



def prepare_traces_categorical(x,y,z, column, values, markersize, opacity, show_unassigned = True, grey_color = '#eeeeee'):
    colormap_discrete = ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7','#000000']
    traces =[]; IDX = []
    if not values:
        values = np.unique(column)
    for roi,i in zip(values, range(len(values))):
        idx = column==roi
        idx = np.where(idx.values==True)[0]
        IDX+=list(idx)
        xx = x[idx]; yy = y[idx]; zz = z[idx]
        traces.append(go.Scatter3d(x=xx,  y=yy, z=zz, mode='markers', marker={'size': markersize, 'opacity': opacity, 'color': colormap_discrete[i%len(colormap_discrete)]},
        name=roi))
    if show_unassigned:
        xx = np.delete(x, IDX); yy = np.delete(y, IDX); zz = np.delete(z, IDX); 
        traces.append(go.Scatter3d(x=xx,  y=yy, z=zz, mode='markers', marker={'size': markersize, 'opacity': opacity, 'color': grey_color}, name="unassigned"))
    return traces
        
    
    
    
def prepare_color_list(column, values, grey_color = '#eeeeee'):
    colormap_discrete = ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7','#000000']
    
    if not values:
        values = np.unique(column)
    color_list = [grey_color]*column.shape[0]  
    i=0
    for roi in values:
        idx = column==roi
        idx = np.where(idx.values==True)[0]
        for idd in idx:
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

def _set_equal_axes(fig, x, y, z):
    import numpy as np

    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()

    # largest span
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if span == 0:
        span = 1.0

    # centers
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2

    # symmetric ranges
    xr = [cx - span/2, cx + span/2]
    yr = [cy - span/2, cy + span/2]
    zr = [cz - span/2, cz + span/2]

    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=xr),
            yaxis=dict(range=yr),
            zaxis=dict(range=zr),
        )
    )

def prepare_trace_continuous(adata, gene_names, x, y, z, cmin, cmax, markersize):
    #hardcoded colors taken from c2l
    final_color_list = [[90, 20, 165], [213, 94, 0], [0, 158, 115], [86, 180, 233], [240, 228, 66], [200, 200, 200], [50, 50, 50]]
    #cycle is meant to mix colors and opacity in case of several genes(sets) to be displayed
    cmin_list = []; cmax_list = []
    for i in range(np.min([len(final_color_list), len(gene_names)])):
        if "SYMBOL" in adata.var.columns:
            id_gene = adata.var["SYMBOL"] == gene_names[i]
            matches = np.where(id_gene)[0]
        else:
            matches = np.where(adata.var_names.astype(str) == str(gene_names[i]))[0]

        if len(matches) == 0:
            raise ValueError(f"Gene '{gene_names[i]}' not found in adata.var['SYMBOL'] or adata.var_names")

        idx = matches[0]

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


def get_xyz(
    adata,
    pixelsize_xy,
    pixelsize_z,
    obsm_positions,
    thickness_sections_table_path=None,
    thickness_one_section=5,
    use_obsm_z=True,
    section_obs_col="section",
    debug=False,
):
    coords = np.asarray(adata.obsm[obsm_positions])
    if coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{obsm_positions}'] must have at least 2 columns"
        )

    has_obsm_z = coords.shape[1] >= 3

    if use_obsm_z and has_obsm_z:
        x, y, z, row_order = create_3D_spot_pos(
            adata,
            obsm_positions,
            use_obsm_z=True,
            section_obs_col=section_obs_col,
            debug=debug,
        )
    elif thickness_sections_table_path:
        thickness_table = pd.read_csv(thickness_sections_table_path)
        x, y, z, row_order = create_3D_spot_pos_thickness_from_table(
            adata,
            obsm_positions,
            thickness_table,
            thickness_one_section,
            use_obsm_z=False,
            section_obs_col=section_obs_col,
            debug=debug,
        )
    else:
        x, y, z, row_order = create_3D_spot_pos(
            adata,
            obsm_positions,
            use_obsm_z=False,
            section_obs_col=section_obs_col,
            debug=debug,
        )

    x = x * pixelsize_xy
    y = y * pixelsize_xy
    z = z * pixelsize_z

    if debug:
        print("final xyz lengths:", len(x), len(y), len(z))
        print("final unique z:", np.unique(z))

    return x, y, z, row_order


def prepare_traces_categorical(
    x, y, z,
    column,
    values,
    markersize,
    opacity,
    show_unassigned=True,
    grey_color='#eeeeee',
    categorical_colors=None
):
    colormap_discrete = ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7','#000000']
    traces =[]; IDX = []
    if not values:
        values = np.unique(column)

    for roi, i in zip(values, range(len(values))):
        idx = column==roi
        idx = np.where(idx.values==True)[0]
        IDX+=list(idx)
        xx = x[idx]; yy = y[idx]; zz = z[idx]

        if categorical_colors is not None and roi in categorical_colors:
            color = categorical_colors[roi]
        else:
            color = colormap_discrete[i % len(colormap_discrete)]

        traces.append(go.Scatter3d(
            x=xx, y=yy, z=zz,
            mode='markers',
            marker={'size': markersize, 'opacity': opacity, 'color': color},
            name=roi
        ))

    if show_unassigned:
        xx = np.delete(x, IDX); yy = np.delete(y, IDX); zz = np.delete(z, IDX)
        traces.append(go.Scatter3d(
            x=xx, y=yy, z=zz,
            mode='markers',
            marker={'size': markersize, 'opacity': opacity, 'color': grey_color},
            name="unassigned"
        ))

    return traces

def plot_3D_interactive_plotly(
    adata,
    obs_column_name=None,
    gene_names=None,
    values=None,
    pixelsize_xy=1,
    pixelsize_z=1,
    units="um",
    markersize=1,
    opacity=0.8,
    obsm_positions="spatial_affine_postreg",
    cmin=None,
    cmax=None,
    colormap="Reds",
    background_black=False,
    thickness_sections_table_path=None,
    save_html=None,
    thickness_one_section=5,
    show_unassigned=True,
    use_obsm_z=True,
    equal_axis=False,
    categorical_colors=None
):
    """
    Plot 3D points from AnnData.

    z handling priority:
    1. If use_obsm_z=True and adata.obsm[obsm_positions] has 3 columns, use col 3 as z.
    2. Else if thickness_sections_table_path is provided, compute z from that CSV.
    3. Else use section order.
    """
    plotly.offline.init_notebook_mode()

    if obs_column_name and gene_names:
        adata_filtered = filter_anndata(adata, obs_column_name, values)
        x, y, z, row_order = get_xyz(
            adata_filtered,
            pixelsize_xy,
            pixelsize_z,
            obsm_positions,
            thickness_sections_table_path,
            thickness_one_section,
            use_obsm_z=use_obsm_z,
        )
        trace, colors_list_cont, cmin_list, cmax_list = prepare_trace_continuous(
            adata_filtered, gene_names, x, y, z, cmin, cmax, markersize
        )
        data = [trace]

    elif obs_column_name and not gene_names:
        x, y, z, row_order = get_xyz(
            adata,
            pixelsize_xy,
            pixelsize_z,
            obsm_positions,
            thickness_sections_table_path,
            thickness_one_section,
            use_obsm_z=use_obsm_z,
        )
        column = adata.obs[obs_column_name].iloc[row_order]
        trace = prepare_traces_categorical(
            x, y, z, column, values, markersize, opacity, show_unassigned,
            categorical_colors=categorical_colors
        )
        data = trace

    elif gene_names and not obs_column_name:
        x, y, z, row_order = get_xyz(
            adata,
            pixelsize_xy,
            pixelsize_z,
            obsm_positions,
            thickness_sections_table_path,
            thickness_one_section,
            use_obsm_z=use_obsm_z,
        )
        trace, colors_list_cont, cmin_list, cmax_list = prepare_trace_continuous(
            adata, gene_names, x, y, z, cmin, cmax, markersize
        )
        data = [trace]

    else:
        raise ValueError("Please specify gene_names or obs_column_name!")

    layout = go.Layout(
        {
            "xaxis": {
                "range": [0.2, 1],
                "showgrid": False,
                "zeroline": False,
                "visible": False,
                "title": str("x, " + units),
            },
            "yaxis": {
                "range": [0.2, 1],
                "showgrid": False,
                "zeroline": False,
                "visible": False,
                "title": "aaaaaa",
            },
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        scene=dict(
            aspectmode="data",
            xaxis=dict(title=dict(text="x, " + units)),
            yaxis=dict(title=dict(text="y, " + units)),
            zaxis=dict(title=dict(text="z, " + units)),
        ),
    )

    plot_figure = go.Figure(data=data, layout=layout)

    if equal_axis:
    # enforce equal units on all axes if asked
        _set_equal_axes(plot_figure, x, y, z)
    if obs_column_name and not gene_names:
        pass
    else:
        add_legend_continuous(
            plot_figure, colors_list_cont, cmin_list, cmax_list, gene_names
        )

    if background_black:
        plot_figure.update_layout(
            paper_bgcolor="rgba(0,0,0,255)",
            font=dict(color="white"),
        )
        plot_figure.update_scenes(
            xaxis=set_bgcolor(),
            yaxis=set_bgcolor(),
            zaxis=set_bgcolor(),
        )

    if save_html:
        plot_figure.write_html(save_html)

    return plot_figure
