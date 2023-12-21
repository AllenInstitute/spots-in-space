import numpy as np
import anndata as ad
import pandas as pd
import pathlib
from scipy.io import mmwrite,mmread
import gzip
import shutil
from zipfile import ZipFile

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