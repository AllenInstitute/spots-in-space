from __future__ import annotations
import numpy as np
import anndata as ad
import pandas as pd
import pathlib
from pathlib import Path
from scipy.io import mmwrite,mmread
import scanpy as sc
import gzip
import shutil
from zipfile import ZipFile
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely import coverage_union_all
import shapely
from matplotlib import pyplot as plt
import seaborn as sns


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


def load_config(configfile: str|Path|None=None):
    """
    loads configuration yaml file for spatial analysis. If no file is found, returns empty dict

    Args:
        configfile: path to yaml file. If None, will look for spatial_config.yml in the directory above this file 
    Returns:
        dict of configuration parameters
    """
    import yaml

    if isinstance(configfile, str):
        configfile = Path(configfile)

    elif configfile is None:
        configfile = Path(__file__).parents[1].joinpath('spatial_config.yml').resolve()
        print(configfile)
        
    if configfile.is_file():
        if hasattr(yaml, 'FullLoader'):
            # pyyaml new API
            config = yaml.load(open(configfile, 'rb'), Loader=yaml.FullLoader)
        else:
            # pyyaml old API
            config = yaml.load(open(configfile, 'rb'))

    else:
        config = {}
    return config


def package_for_10x(anndata_object,
                    output_directory,
                    gene_id_var_list, 
                    dry_run = False,
                    exist_ok =False,
                   annotation_category="Supertype"):
    """
    takes a reference dataset as anndata object and writes a reference file that can be uploaded
    to 10x as a reference for  Xenium  gene panel selection
    see https://www.10xgenomics.com/support/in-situ-gene-expression/documentation/steps/panel-design/xenium-panel-getting-started#input-ref-anno


    keyword arguments dry_run and exist_ok may help you avoid overwriting something important
    """
    # organize some paths:
    pathlib.Path(output_directory).mkdir(exist_ok=exist_ok)
    zip_path = pathlib.Path(output_directory).parent.joinpath(output_directory.stem+"_to_zip")
    zip_path.mkdir(exist_ok = exist_ok)
    matrix_output_path = output_directory.joinpath( "matrix.mtx")
    barcodes_output_path = output_directory.joinpath( "barcodes.tsv")
    features_output_path = output_directory.joinpath( "features.tsv" )
    annotation_output_path =output_directory.joinpath("annotation.csv")
    # Going for MEX format here:

    # confirmed this matches after reading it back in, although the read in is float64

    # barcodes.tsv


    # features.tsv.gz
    # The file is expected to conform to the specification outlined under MEX format here, namely:

    #     Tab delimited
    #     No header column
    #     Ensembl IDs, followed by gene symbols, optionally followed by a feature type
    # 
    # ENSG00000141510       TP53         Gene Expression
    # ENSG00000012048       BRCA1        Gene Expression
    # ENSG00000139687       RB1          Gene Expression



    features_tsv = pd.DataFrame(anndata_object.var_names, columns=["Gene Symbol"])
    features_tsv["Ensembl IDs"] = gene_id_var_list
    features_tsv = features_tsv.loc[:,["Ensembl IDs","Gene Symbol"]]
    # annotations.csv  needs #barcode column and #annotation column:
    tout_annotations = anndata_object.obs.copy()
    tout_annotations["barcode"]= tout_annotations.index.values
    tout_annotations["annotation"]= tout_annotations[annotation_category].values
    
    #actual writing to files:
    
    if not dry_run:
        # matrix.mtx.gz
        mmwrite(matrix_output_path, anndata_object.layers["UMIs"].T)

        pd.DataFrame(anndata_object.obs.index).to_csv(barcodes_output_path, sep = "\t", index = False, header=False)
        features_tsv.to_csv(features_output_path,   sep = "\t", index=False, header=False)
        tout_annotations.loc[:,["barcode","annotation"]].to_csv(annotation_output_path,index=False)
        
        
        # compress individual files:
        files_in_target = []
        for file in [barcodes_output_path, features_output_path, matrix_output_path]:
            file_in_target = str(zip_path.joinpath(file.name))+".gz"
            with open(file,'rb') as to_zip:
                with gzip.open(file_in_target, 'wb') as zip_out:
                    shutil.copyfileobj(to_zip, zip_out)
                    files_in_target.append(file_in_target)
        # copy over the annotation:
        shutil.copyfile(annotation_output_path, str(zip_path.joinpath(annotation_output_path.name)))
        files_in_target.append(str(zip_path.joinpath(annotation_output_path.name)))
        # then zip the whole directory:
        with ZipFile(str(zip_path)+".zip", 'w') as final_zip:
            for f in files_in_target:
                final_zip.write(f, arcname = pathlib.Path(f).name)
                
        
    else:
        return features_tsv

def plot_genes(spottable,  gene_list, 
               min_counts,highlight_list = [], 
               subsample=1,
               figsize=[15,15], 
               transpose_plot = True, fontsize=20, color_background =[.65,.65,.65],
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
                    plt.plot(toplotx, toploty, 'x', color = color_background,markersize=markersize_background, label = "other_genes")
                    first_background = False
                else:
                    plt.plot(toplotx, toploty, 'x', color = color_background,markersize=markersize_background, label = None)
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
                               cell_annotation_values_background = None,
                               cell_annotation_colormapping= None,
                               plot_blanks =False,
                               loaded_image_array = None,
                               loaded_image_extent = None,
                               initial_figsize=[20,20],
                               fontsize=20, image_cmap="Greys",
                               selected_cell_outline_weight = 1.0,
                               plot_patches=False,skip_genes=False,
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

    if not skip_genes:
        plot_genes(spottable,gene_list, 1,
               highlight_list =genes_to_highlight,figsize=initial_figsize, incoming_ax=ax, **kwargs)
    plt.axis('equal')
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





    if cell_annotation_colormapping is None:
        plotted_categories={pc:{"plotted":False,
                                "color":no_gray2[ii],
                                "linewidth":selected_cell_outline_weight,
                               "edgecolor":'k'} for ii,pc in enumerate(cell_annotation_values)}
    else:
        plotted_categories={pc:{"plotted":False,
                                "color":cell_annotation_colormapping[pc][0],
                               "linewidth":cell_annotation_colormapping[pc][1],
                               "edgecolor":cell_annotation_colormapping[pc][2]}
                            for ii,pc in enumerate(list(anno[cell_annotation_category].unique()))}


    if type(segmentation_geopandas) == gpd.geodataframe.GeoDataFrame :


        for cellid in segmentation_geopandas.loc[segmentation_geopandas.EntityID.isin(spottable.cell_ids),["EntityID","Geometry"]].EntityID.unique():
            cellinfo = segmentation_geopandas.loc[segmentation_geopandas.EntityID==cellid,"Geometry"].values[0]
            tg = coverage_union_all(cellinfo)
            try:
                if  isinstance(tg, shapely.Polygon):
                    x_coords = list(tg.boundary.coords.xy[1])
                    y_coords = list(tg.boundary.coords.xy[0])
                elif isinstance(tg, shapely.MultiPolygon):
                    # these are multiple polygons. take the largest. TODO: find out how this happens...
                    geo_list = [g for g in tg.geoms]
                    largest = np.argmax([g.area for g in geo_list])
                    x_coords = list(geo_list[largest].boundary.coords.xy[1])
                    y_coords = list(geo_list[largest].boundary.coords.xy[0])                
                else:
                    print(str(cellid)+"   is neither Polygon nor MultiPolygon")
                    continue
            except:
                print(str(cellid)+"   failed decomposition")
                continue               
  
            if cellid in list(anno.loc[anno[cell_annotation_category].isin(cell_annotation_values),"cell_id"]):
                for ii,anno_value in enumerate(cell_annotation_values):
                    anno_list = list(anno.loc[anno[cell_annotation_category]==anno_value,"cell_id"])
                    if cellid in anno_list :
                        if not plotted_categories[anno_value]["plotted"]:
                            # first time through, include label for legend
                            if plot_patches:
                                plt.fill(x_coords, y_coords, 
                                    facecolor=plotted_categories[anno_value]["color"],
                                         linewidth=plotted_categories[anno_value]["linewidth"],
                                         edgecolor=plotted_categories[anno_value]["edgecolor"],
                                         label = anno_value)
                                # with thin boundary
                                plt.plot(x_coords, y_coords,
                                         color=[.2,.2,.2],linewidth=.5)
                            else:
                                plt.plot(x_coords, y_coords, 
                                    color=plotted_categories[anno_value]["color"],
                                    linewidth=plotted_categories[anno_value]["linewidth"],
                                    label = anno_value)
                            plotted_categories[anno_value]["plotted"]=True
                        else:
                            # other times through, do not include label for legend
                            if plot_patches:
                                plt.fill(x_coords, y_coords, 
                                 facecolor=plotted_categories[anno_value]["color"],
                                         linewidth=plotted_categories[anno_value]["linewidth"],
                                         edgecolor=plotted_categories[anno_value]["edgecolor"],
                                         label = None)
                                # with added thin boundary
                                plt.plot(x_coords, y_coords,
                                         color=[.2,.2,.2],linewidth=1)
                            else:

                                plt.plot(x_coords, y_coords, 
                                 color=plotted_categories[anno_value]["color"],
                                         linewidth=plotted_categories[anno_value]["linewidth"],
                                         label = None)


            elif cellid in list(anno.loc[anno[cell_annotation_category].isin(cell_annotation_values_background),"cell_id"]):
                for ii,anno_value in enumerate(cell_annotation_values_background):
                    anno_list = list(anno.loc[anno[cell_annotation_category]==anno_value,"cell_id"])
                    if cellid in anno_list :
                        if not plotted_categories[anno_value]["plotted"]:
                            # first time through, include label for legend
                            if plot_patches:
                                plt.fill(x_coords, y_coords, 
                                    facecolor=plotted_categories[anno_value]["color"],
                                         linewidth=0,
                                         edgecolor=plotted_categories[anno_value]["edgecolor"],
                                         label = anno_value)

                            else:
                                plt.plot(x_coords, y_coords, 
                                    color=plotted_categories[anno_value]["color"],
                                    linewidth=plotted_categories[anno_value]["linewidth"],
                                    label = anno_value)
                            plotted_categories[anno_value]["plotted"]=True
                        else:
                            # other times through, do not include label for legend
                            if plot_patches:
                                plt.fill(x_coords, y_coords, 
                                 facecolor=plotted_categories[anno_value]["color"],
                                         linewidth=0,
                                         edgecolor=plotted_categories[anno_value]["edgecolor"],
                                         label = None)

                            else:

                                plt.plot(x_coords, y_coords, 
                                 color=plotted_categories[anno_value]["color"],
                                         linewidth=plotted_categories[anno_value]["linewidth"],
                                         label = None)


            #add thin outlines to everything?

            else:
                plt.plot(x_coords, y_coords,
                         color=[.2,.2,.2],linewidth=.5)
#             except:
#                 print("skipping plot of "+str(cellid))
    plt.legend()
    # could be useful at some point: get polygons from spotdata:
    # for k in list(mini.cell_polygons.keys()):
    #     if mini.cell_polygons[k] == None:
    #         continue
    #     if mini.cell_polygons[k].boundary == None:
    #         continue
    #     plt.plot(list(mini.cell_polygons[k].boundary.coords.xy[1]), list(mini.cell_polygons[k].boundary.coords.xy[0]))


    
def plot_cbg_centroids(cell_by_gene: ad.AnnData, ax, x='center_x', y='center_y', **kwargs):
    g = sns.scatterplot(data=cell_by_gene.obs, x=x, y=y, ax=ax, **kwargs)
    ax.set_aspect('equal', adjustable='box', anchor='C')
    return g 


def make_cirro_compatible(cell_by_gene: ad.AnnData):
    '''Make an AnnData object compatible with Cirrocumulus visualization tool.
    
    Copy cell spatial coordinates to obsm, as expected by Cirrocumulus
    (https://cirrocumulus.readthedocs.io). Also, generate UMAP from .X as an 
    additional visualization option.

    Parameters:
        cell_by_gene: AnnData object output by 
                      sis.spot_table.cell_by_gene_anndata()

    Returns:
        cell_by_gene_cirro: copy of cell_by_gene with cirrocumulus-compatible fields
    '''
    # confirm AnnData is in correct format
    assert_message = 'cell_by_gene obs columns must match format output by sis.spot_table.cell_by_gene_anndata()' 
    assert {'center_x','center_y','center_z'}.issubset(cell_by_gene.obs.columns), assert_message

    cell_by_gene_cirro = cell_by_gene.copy()

    # Copy cell coordinates to obsm
    cell_by_gene_cirro.obsm['spatial'] = cell_by_gene_cirro.obs[
                                                                ['center_x', 
                                                                'center_y', 
                                                                'center_z']
                                                                ].to_numpy()
    
    # Generate UMAP of X as alternative visualization
    # already cirro-compatible as it's stored in obsm['X_umap']
    sc.pp.pca(cell_by_gene_cirro)
    sc.pp.neighbors(cell_by_gene_cirro)
    sc.tl.umap(cell_by_gene_cirro)
    # Optional future update: UMAP of all layers (will run much slower)
    # for layer in cell_by_gene.layers.keys():
    #     sc.pp.pca(cell_by_gene, layer=layer)
    #     sc.pp.neighbors(cell_by_gene, layer=layer)
    #     sc.tl.umap(cell_by_gene, layer=layer)

    return cell_by_gene_cirro
