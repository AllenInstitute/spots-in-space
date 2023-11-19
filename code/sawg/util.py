import numpy as np
from matplotlib import pyplot as plt

import geopandas as gpd
import pandas as pd
from shapely import coverage_union_all
import anndata as ad




def reduce_expression(data, umap_args):
    import umap
    from sklearn.preprocessing import StandardScaler

    default_umap_args = {'n_neighbors': 3, 'min_dist': 0.4, 'n_components': 3}
    default_umap_args.update(umap_args)

    flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    # randomize order (because umap has some order-dependent effects)
    order = np.arange(flat_data.shape[0])
    np.random.shuffle(order)
    flat_data = flat_data[order]

    # remove rows with no transcripts
    mask = flat_data.sum(axis=1) > 0
    masked_data = flat_data[mask]

    # scale in prep for umap
    scaler = StandardScaler()
    scaled = scaler.fit_transform(masked_data)

    # reduce down to 3D
    reducer = umap.UMAP(**default_umap_args)
    reduced = reducer.fit_transform(scaled)

    # re-insert rows with no transcripts (all 0)
    final = np.zeros((len(flat_data), reduced.shape[1]), dtype=reduced.dtype)
    final[mask] = reduced

    # un-shuffle order
    reverse_order = np.argsort(order)
    final = final[reverse_order]

    # return reshaped to original image
    return final.reshape(data.shape[0], data.shape[1], final.shape[-1])


def map_to_ubyte(data):
    mn, mx = data.min(), data.max()
    return np.clip((data - mn) * 255 / (mx - mn), 0, 255).astype('ubyte')


def rainbow_wheel(points, center=None, radius=None, center_color=None):
    """Given an Nx2 array of point locations, return an Nx3 array of RGB
    colors derived from a rainbow color wheel centered over the mean point location.
    """
    import matplotlib.pyplot as plt
    import scipy.interpolate
    flat = points.reshape(np.product(points.shape[:-1]), points.shape[-1])
    if center is None:
        center = flat.mean(axis=0)
    if radius is None:
        radius = 4 * flat.std(axis=0)
    f = np.linspace(0, 1, 10)[:-1]
    theta = f * 2 * np.pi
    x = np.vstack([radius[0] * np.cos(theta) + center[0], radius[1] * np.sin(theta) + center[1]]).T
    c = plt.cm.gist_rainbow(f)[:, :3]

    if center_color is not None:
        x = np.concatenate([x, center[None, :]], axis=0)
        c = np.concatenate([c, np.array(center_color)[None, :]], axis=0)

    color = scipy.interpolate.griddata(x, c, flat[:, :2], fill_value=0)
    return color.reshape(points.shape[:-1] + (3,))


def show_float_rgb(data, extent, ax):
    """Show a color image given a WxHx3 array of floats.
    Each channel is normalized independently.
    """
    rgb = np.empty(data.shape[:2] + (3,), dtype='ubyte')
    for i in (0, 1, 2):
        rgb[..., i] = map_to_ubyte(data[..., i])

    return ax.imshow(rgb, extent=extent, aspect='equal', origin='lower')


def log_plus_1(x):
    return np.log(x + 1)


def poly_to_geojson(polygon):
    """
    turns a single shapely Polygon into a geojson polygon
    Args:
        polygon shapely.Polygon
    Returns:
        geojson polygon
    """
    import geojson
    poly_array = np.array(polygon.exterior.coords)

    return geojson.Polygon([[(poly_array[i,0], poly_array[i,1]) for i in range(poly_array.shape[0])]])

def plot_genes(spottable,  gene_list, 
               min_counts,highlight_list = [], 
               subsample=1,
               figsize=[15,15], 
               transpose_plot = True, fontsize=20, 
               markersize_background = 1.5, markersize_highlight=5,
               color_start =0, incoming_ax = None ):
    
    
    first_background = True
    no_gray = list(plt.cm.tab10.colors[:7])
    no_gray.extend(plt.cm.tab10.colors[8:])
    no_gray = list(np.roll(np.array(no_gray),[color_start,0], axis = [0,1]))
    
    if incoming_ax:
        ax= incoming_ax
    else:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    
    ax.set_prop_cycle('color', no_gray)

    for g in gene_list:
        gmask = spottable.gene_names==g

        if np.sum(gmask) >min_counts:
            if g not in highlight_list:
                if transpose_plot:
                    toploty = spottable.x[gmask][::subsample]
                    toplotx = spottable.y[gmask][::subsample]
                else:
                    toplotx = spottable.x[gmask][::subsample]
                    toploty = spottable.y[gmask][::subsample]                   
                if first_background:
                    plt.plot(toplotx, toploty, 'x', color = [.65,.65,.65],markersize=markersize_background, label = "other_genes")
                    first_background = False
                else:
                    plt.plot(toplotx, toploty, 'x', color = [.65,.65,.65],markersize=markersize_background, label = None)
    for g in gene_list:
        gmask = spottable.gene_names==g

        if np.sum(gmask) >min_counts:
            if g in highlight_list:
                if transpose_plot:
                    toploty = spottable.x[gmask][::subsample]
                    toplotx = spottable.y[gmask][::subsample]
                else:
                    toplotx = spottable.x[gmask][::subsample]
                    toploty = spottable.y[gmask][::subsample] 
                plt.plot(toplotx, toploty, '.', label = g, markersize=markersize_highlight)

    plt.axis('equal')
    plt.legend(fontsize=fontsize, markerscale=4)



    
def show_cells_and_transcripts(spottable, anndata_obj,
                               segmentation_geopandas,
                               genes_to_highlight=[],
                               cell_annotation_category = "supertype_scANVI_leiden",
                               cell_annotation_values = None,
                               plot_blanks =False,
                               loaded_image_array = None,
                               loaded_image_extent = None,
                               initial_figsize=[20,20],
                               cell_annotation_colors = ['k'],fontsize=20, image_cmap="Greys",
                               selected_cell_outline_weight = 1.0,
                               **kwargs):
    """
    Parameters:
    
    loaded_image_array
    
    will take a 2D numpy array and use it for background. otherwise a maximum projection of image data from the SpotTable is created and used.
    
    cell_annotation_category
    
    column in anndata_obj.obs to get annotation information from
    
    
    cell_annotation_values 
    
    values in `cell_annotation_category` to show
    
    
    **kwargs are passed to `plot_genes`
    
    """
    

    no_gray2 = list(plt.cm.tab10.colors[1:7])
    no_gray2.extend(plt.cm.tab10.colors[8:])
    np.roll(np.array(no_gray2),[2,0], [0,1])




    # get image data and show it.
    plot_image = False
    if isinstance(loaded_image_array,type(None)):

        spot_table_im =spottable.get_image(channel="DAPI")

        image_data = spot_table_im.get_data()
        # return max projection over z, transposed and flipped vertically for display
        loaded_image_array = np.max(image_data,axis=0).T[::-1,:]
        loaded_image_extent = np.array(spot_table_im.bounds())[::-1,:].ravel()
        plot_image = True
    else:
        # check input types pls
        pass

    
    fig = plt.figure(figsize=initial_figsize)
    if plot_image:
        plt.imshow(loaded_image_array, extent=loaded_image_extent, cmap=image_cmap, vmax = 0.8*np.max(loaded_image_array))   

    ax = plt.gca()
    targets = np.unique(spottable.gene_names)

    if plot_blanks:
        gene_list = list(targets)
    else:
        gene_list = [g for g in targets if "Blank-" not in g]


    plot_genes(spottable,gene_list, 1,
               highlight_list =genes_to_highlight,figsize=initial_figsize, incoming_ax=ax, **kwargs)
    plt.gca().invert_yaxis()



    # plot cell perimeters and gather centroids along the way
    # this particular case (2D segmentation on 3d data) means that we have 7 copies of each segmentation polygon (and 7 centroids for each cell)
    # to deal with this, I'm going to take only the 0th polygon for each unique cell id.
    # in the case where there is full 3D segmentation, it should show up plotted below and the centroids should still be reasonably accurate






    # add mapped cell identities:
    # suboptimal copy here
    if type(anndata_obj) == ad.AnnData:
        anno = anndata_obj.obs.copy()
        anno["cell_id"] = anno.index.values.astype(int)   






    plotted_categories={pc:{"plotted":False,"color":no_gray2[ii]} for ii,pc in enumerate(cell_annotation_values)}



    if type(segmentation_geopandas) == gpd.geodataframe.GeoDataFrame :


        for cellid in segmentation_geopandas.loc[segmentation_geopandas.EntityID.isin(spottable.cell_ids),["EntityID","Geometry"]].EntityID.unique():
            cellinfo = segmentation_geopandas.loc[segmentation_geopandas.EntityID==cellid,"Geometry"].values[0]
            tg = coverage_union_all(cellinfo)
            try:
                if cellid in list(anno.loc[anno[cell_annotation_category].isin(cell_annotation_values),"cell_id"]):
                    for ii,anno_value in enumerate(cell_annotation_values):
                        anno_list = list(anno.loc[anno[cell_annotation_category]==anno_value,"cell_id"])
                        if cellid in anno_list :
                            if not plotted_categories[anno_value]["plotted"]:
                                plt.plot(list(tg.boundary.coords.xy[1]), list(tg.boundary.coords.xy[0]), 
                                 color=plotted_categories[anno_value]["color"],linewidth=selected_cell_outline_weight,
                                         label = anno_value)
                                plotted_categories[anno_value]["plotted"]=True
                            else:
                                plt.plot(list(tg.boundary.coords.xy[1]), list(tg.boundary.coords.xy[0]), 
                                 color=plotted_categories[anno_value]["color"],linewidth=selected_cell_outline_weight,
                                         label = None)

                else:
                    plt.plot(list(tg.boundary.coords.xy[1]), list(tg.boundary.coords.xy[0]),
                             color=[.2,.2,.2],linewidth=.5)
            except:
                print("skipping plot of "+str(cellid))
    plt.legend()
    # could be useful at some point: get polygons from spotdata:
    # for k in list(mini.cell_polygons.keys()):
    #     if mini.cell_polygons[k] == None:
    #         continue
    #     if mini.cell_polygons[k].boundary == None:
    #         continue
    #     plt.plot(list(mini.cell_polygons[k].boundary.coords.xy[1]), list(mini.cell_polygons[k].boundary.coords.xy[0]))


    
 