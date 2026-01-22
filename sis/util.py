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
from shapely import coverage_union_all
import shapely
from matplotlib import pyplot as plt
import seaborn as sns

import zipfile

from .spot_table import SpotTable

def reduce_expression(data, umap_args):
    """Reduce the expression data using UMAP.

    Parameters
    ----------
    data : np.ndarray
        The input expression data.
    umap_args : dict
        Additional UMAP arguments.
   
    Returns
    -------
    ndarray
        The reduced expression data.
    """
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
    """Map a 2D or 3D array of floats to an 8-bit unsigned integer array.
    
    Parameters
    ----------
    data : np.ndarray
        The input array of floats.
    
    Returns
    -------
    np.ndarray
        The mapped array with values between 0 and 255.
    """
    mn, mx = data.min(), data.max()
    return np.clip((data - mn) * 255 / (mx - mn), 0, 255).astype('ubyte')


def rainbow_wheel(points, center=None, radius=None, center_color=None):
    """Given an Nx2 array of point locations, return an Nx3 array of RGB
    colors derived from a rainbow color wheel centered over the mean point location.
    
    Parameters
    ----------
    points : np.ndarray
        Nx2 array of point locations
    center : np.ndarray or None, optional
        1x2 array of center location (default: mean of points)
    radius : np.ndarray or None, optional
        1x2 array of radius in x and y directions (default: 4 * std of points)
    center_color : tuple or None, optional
        RGB color for the center point
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
    
    Parameters
    ----------
    data : np.ndarray
        WxHx3 array of floats
    extent : list
        List of four floats defining the extent of the image (see matplotlib.pyplot.imshow)
    ax : matplotlib.axes.Axes
        Axes to plot the image on
        
    Returns
    -------
    matplotlib.image.AxesImage
        The image displayed on the axes
    """
    rgb = np.empty(data.shape[:2] + (3,), dtype='ubyte')
    for i in (0, 1, 2):
        rgb[..., i] = map_to_ubyte(data[..., i])

    return ax.imshow(rgb, extent=extent, aspect='equal', origin='lower')


def load_config(configfile: str|Path|None=None):
    """loads configuration yaml file for spatial analysis. 

    Parameters
    ----------
    configfile : str or Path or None, optional
        Path to yaml file. If None, will look for spatial_config.yml in the directory above this file 
    
    Returns
    -------
    dict
        Dict of configuration parameters. If no file is found, returns empty dict
    """
    import yaml

    if isinstance(configfile, str):
        configfile = Path(configfile)

    elif configfile is None:
        configfile = Path(__file__).parents[1].joinpath('spatial_config.yml').resolve()
        
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
    Takes a reference dataset as anndata object and writes a reference file that can be uploaded
    to 10x as a reference for  Xenium  gene panel selection
    see https://www.10xgenomics.com/support/in-situ-gene-expression/documentation/steps/panel-design/xenium-panel-getting-started#input-ref-anno


    keyword arguments dry_run and exist_ok may help you avoid overwriting something important
    
    Parameters
    ----------
    anndata_object : AnnData
        The input AnnData object.
    output_directory : Path
        The directory to save the output files.
    gene_id_var_list : list
        List of gene IDs for the features.tsv file.
    dry_run : bool, optional
        If True, only return features_tsv without writing files. Default is False.
    exist_ok : bool, optional
        If True, do not raise an error if the output directory already exists. Default is False.
    annotation_category : str, optional
        Column in anndata_object.obs to use for annotations. Default is "Supertype".
        
    Returns
    -------
    pandas.DataFrame
        If dry_run is True, returns features_tsv DataFrame. 
    None 
        If dry_run is False, writes files to output_directory.
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
    """Plot the locations of genes in a SpotTable.
    
    Parameters
    ----------
    spottable : SpotTable
        The SpotTable object containing gene expression data.
    gene_list : list
        List of genes to plot.
    min_counts : int
        Minimum number of counts for a gene to be plotted.
    highlight_list : list, optional
        List of genes to highlight. Default is empty list.
    subsample : int, optional
        Subsampling factor for plotting. Default is 1 (no subsampling).
    figsize : list, optional
        Figure size. Default is [15, 15].
    transpose_plot : bool, optional
        Whether to transpose the plot. Default is True.
    fontsize : int, optional
        Font size for the legend. Default is 20.
    color_background : list, optional
        Color for background points. Default is [.65, .65, .65].
    markersize_background : float, optional
        Marker size for background points. Default is 1.5.
    markersize_highlight : float, optional
        Marker size for highlighted points. Default is 5.
    color_start : int, optional
        Starting index for color cycle. Default is 0.
    incoming_ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes will be created. Default is None.
    """
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
    Parameters
    ----------
    spottable : SpotTable
        The SpotTable object containing gene expression data.
    anndata_obj : AnnData
        The AnnData object containing cell annotations.
    segmentation_geopandas : geopandas.GeoDataFrame
        GeoDataFrame containing cell segmentation polygons.
    genes_to_highlight : list, optional
        List of genes to highlight in the plot. Default is empty list.
    cell_annotation_category : str, optional
        column in anndata_obj.obs to get annotation information from
    cell_annotation_values : list or None, optional
        List of values in `cell_annotation_category` to show in the plot.
    cell_annotation_values_background : list or None, optional
        List of values in `cell_annotation_category` to show in the background.
    cell_annotation_colormapping : dict or None, optional
        Dictionary mapping cell annotation values to colors and linewidths.
    plot_blanks : bool, optional
        Whether to plot blank genes. Default is False.
    loaded_image_array : np.ndarray or None, optional
        will take a 2D numpy array and use it for background. otherwise a maximum projection of image data from the SpotTable is created and used.
    loaded_image_extent : np.ndarray or None, optional
        Extent of the image data. If not provided, it will be calculated from the SpotTable.
    initial_figsize : list, optional
        Initial figure size for the plot. Default is [20, 20].
    fontsize : int, optional
        Font size for the legend. Default is 20.
    image_cmap : str, optional
        Colormap for the image background. Default is "Greys".
    selected_cell_outline_weight : float, optional
        Line width for selected cell outlines. Default is 1.0.
    plot_patches : bool, optional
        Whether to plot cell outlines as patches. Default is False.
    skip_genes : bool, optional
        Whether to skip plotting genes. Default is False.
    **kwargs
        Pass to plot_genes()
    """
    import geopandas as gpd

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


    
def plot_cbg_centroids(cell_by_gene: ad.AnnData, ax, x='x', y='y', **kwargs):
    """
    Plot the centroids of cells in a cell-by-gene AnnData object.
    
    Parameters
    ----------
    cell_by_gene : AnnData
        The input AnnData object.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : str, optional
        Column name for x-coordinates. Default is 'x'.
    y : str, optional
        Column name for y-coordinates. Default is 'y'.
    **kwargs : keyword arguments
        Additional arguments passed to sns.scatterplot.
    """
    g = sns.scatterplot(data=cell_by_gene.obs, x=x, y=y, ax=ax, **kwargs)
    ax.set_aspect('equal', adjustable='box', anchor='C')
    return g 


def example_function():
    return 2


def unpack_test_data():
    """Unpacks the test data for the spots-in-space project.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the confirmation strings for the Xenium and Merscope test data.
    """
    SIS_DIR = pathlib.Path().absolute()
    print(SIS_DIR)

    XENIUM_STEM = "xenium_test"
    MERSCOPE_STEM = "merscope_test"

    TEST_DIR = SIS_DIR.joinpath("tests").joinpath("spatial_test_data")
    XENIUM_DIR = TEST_DIR.joinpath(XENIUM_STEM)
    MERSCOPE_DIR = TEST_DIR.joinpath(MERSCOPE_STEM)

    XENIUM_DIR.mkdir(exist_ok = True)
    MERSCOPE_DIR.mkdir(exist_ok = True)


    
    with zipfile.ZipFile(TEST_DIR.joinpath(XENIUM_STEM+".zip")) as z:
        z.extractall(XENIUM_DIR)

    with zipfile.ZipFile(TEST_DIR.joinpath(MERSCOPE_STEM+".zip")) as z:
        z.extractall(MERSCOPE_DIR)


    xenium_confirmation = list(XENIUM_DIR.glob("*"))[0].stem
    merscope_confirmation = list(list(MERSCOPE_DIR.glob("*"))[0].glob("*.vzg"))[0].stem

    return (xenium_confirmation, merscope_confirmation)




def make_cirro_compatible(cell_by_gene: ad.AnnData,obs_spatial_columns =  ['x',  'y'],
                           in_place: bool = False, 
                           generate_umap: bool = True):
    '''Make an AnnData object compatible with Cirrocumulus visualization tool.
    
    Copy cell spatial coordinates to obsm, as expected by Cirrocumulus
    (https://cirrocumulus.readthedocs.io). Also, generate UMAP from .X as an 
    additional visualization option.

    Parameters
    ----------
    cell_by_gene: AnnData object output by 
                    sis.spot_table.cell_by_gene_anndata()
    in_place: bool, if True, modifies the input anndata object in place.
    include_z: bool, whether to include z-coordinate in obsm['spatial'].
                    Default is False.  If True, expects 'center_z' in cell_by_gene.obs.
    generate_umap: bool, whether to generate UMAP from .X. Default is True.
        
    Returns
    -------
    bool 
        If in_place is True, modifies the input AnnData object and returns True.
    cell_by_gene_cirro : anndata.AnnData
        If in_place is False, returns a new AnnData object with the obsm updated to make cirrocumulus-compatible fields.
    '''
    # confirm AnnData is in correct format
    if not set(obs_spatial_columns).issubset(cell_by_gene.obs.columns):
        raise ValueError(f"Columns {obs_spatial_columns} not found in cell_by_gene.obs")




    new_obsm = {}
    # Copy cell coordinates to obsm
    new_obsm['spatial'] = cell_by_gene.obs[obs_spatial_columns].to_numpy()
    

    if generate_umap:
            # Generate UMAP of X as alternative visualization
            # already cirro-compatible as it's stored in obsm['X_umap']
            sc.pp.pca(cell_by_gene)
            sc.pp.neighbors(cell_by_gene)
            sc.tl.umap(cell_by_gene)
            # Optional future update: UMAP of all layers (will run much slower)
            # for layer in cell_by_gene.layers.keys():
            #     sc.pp.pca(cell_by_gene, layer=layer)
            #     sc.pp.neighbors(cell_by_gene, layer=layer)
            #     sc.tl.umap(cell_by_gene, layer=layer)



    if in_place:
            
        cell_by_gene.obsm.update(new_obsm)
        return True

    else:
        cell_by_gene_cirro =cell_by_gene.copy()
        cell_by_gene_cirro.obsm.update(new_obsm)    
        return cell_by_gene_cirro


def convert_value_nested_dict(d: dict, oldtype, newtype) -> dict:
    """Helper function to convert a value in a nested dict from oldtype to newtype.
    
    Parameters
    ----------
    d : dict
        The input dictionary.
    oldtype : type
        The type of the value to be converted.
    newtype : type
        The type to convert the value to.
        
    Returns
    -------
    x : dict
        The dictionary with converted values.
    """
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_value_nested_dict(v, oldtype, newtype)
        elif isinstance(v, oldtype):
            v = newtype(v)
        x[k] = v
    return x

def get_cell_cmap(seg_spot_table, bg_color: str|None = None, remove_negatives: bool = True):
    """Helper function specifically for plotting masks in the segmentation demos
    
    Parameters
    ----------
    seg_spot_table : sis.spot_table.SegmentedSpotTable
        spot table to get cell ids from
    bg_color : str or None, optional
        Color for background cells (0 and -1). If None, no black is set. Default is None.
    remove_negatives : bool, optional
        By default, the SIS cell palette includes more negative values than -1.
        These can throw off visualization, so by default they are removed. Default is True.
        
    Returns
    -------
    cell_cmap : matplotlib.colors.ListedColormap
        cmap value to use for plotting
    """
    from matplotlib import colors
    
    # Create colormap for cell ids
    cell_colors = seg_spot_table.cell_palette(seg_spot_table.cell_ids)

    if remove_negatives:
        # Remove the negative values in cell_palette, which can throw off visualization
        cell_colors = {cell: color for cell, color in cell_colors.items() if cell in seg_spot_table.unique_cell_ids}

    if bg_color is None:
        cell_colors[0] = colors.to_rgba('black')
    else:
        cell_colors[0] = colors.to_rgba(bg_color)
        
    cell_cmap = colors.ListedColormap(dict(sorted(cell_colors.items())).values())

    return cell_cmap

def parse_polygon_geodataframe(gdf: gpd.GeoDataFrame, spot_table: SpotTable, cell_id_col: str='id', z_plane_col: str='z_plane'):
    """Parse a geopandas GeoDataFrame to extract polygon geometries and put them into a SIS polygon dict.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing polygon geometries and associated cell IDs.
    spot_table : SpotTable
        The spottable that the polygons are associated with.
    cell_id_col : str, optional
        The name of the column in the GeoDataFrame that contains cell IDs (default is 'id').
    z_plane_col : str, optional
        The name of the column in the GeoDataFrame that contains z-plane values (default is 'z_plane').

    Returns
    -------
    dict
        A dictionary where keys are cell IDs and values are either None, polygons, or a dictionary mapping z-plane values to polygons.
    """
    # the 'z_plane' and 'id' column names are used in SIS generated dataframes
    if cell_id_col in gdf.columns: # If the cell ids are not set 
        gdf = gdf.set_index(cell_id_col)

    # Need to distinguish cell_labels and cell_ids
    try:
        gdf.index = gdf.index.astype(int)
        id_type = int
    except ValueError:
        id_type = str
        
    if z_plane_col not in gdf.columns:
        geom_dict = gdf['geometry'].to_dict()
        
        # loop over the cells in the spot table and add their polygons a SIS polygon dict
        # cell ids are converted to the appropriate type to query the geometry dict
        # if the value doesn't exist in the geometry dict, we set it to None
        result = {cid: geom_dict.get(spot_table.convert_cell_id(cid) if id_type == str else cid, None) for cid in spot_table.unique_cell_ids}
    else:
        result = {}
        # we loop over the dataframe not a dictionary beceause the cell ids are not unique
        for cid, z_plane, polygon in zip(gdf.index, gdf[z_plane_col], gdf['geometry']):
            cid = spot_table.convert_cell_id(cid) if id_type == str else cid
            if z_plane is None:
                result[cid] = None
            else:
                result.setdefault(int(cid), {})[float(z_plane)] = polygon
    # We add None values for cells without polygons to be consistent with SIS logic
    for cid in spot_table.unique_cell_ids:
        if cid not in result:
            result[cid] = None
    return result