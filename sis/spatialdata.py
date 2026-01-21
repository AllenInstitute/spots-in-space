from pathlib import Path
from types import MappingProxyType
from typing import Literal, Callable, Mapping, Any

import geojson
import geopandas as gpd
import anndata
import tifffile
import numpy as np
import xml.etree.ElementTree as ET
from shapely import MultiPolygon
from xarray import concat, DataTree


import dask
dask.config.set({"dataframe.query-planning": False})
import dask.array as da
import dask.dataframe as dd
from dask_image.imread import imread

from spatialdata import SpatialData
from spatialdata_io.readers.merscope import _get_channel_names
from spatialdata.transformations import Affine, Identity, BaseTransformation
from spatialdata.models import Image2DModel, Image3DModel, ShapesModel, PointsModel, TableModel

from .spot_table import SpotTable


def _is_supported_transformation(transformation: BaseTransformation):
    """Checks if a spatialdata transformation is supported by SIS.
    We only support identity, scaling, and translation transformations

    We do not support rotation or shear at this time.
    We keep all SIS computations in transcript coordinate space. Thus, when an image is not rotationally aligned we don't have an easy way to subset its pixels for plotting
    This is a solvable problem, but out of scope for now.

    Parameters
    ----------
    transformation : BaseTransformation
        A spatialdata transformation to check
    
    Returns
    -------
    bool
    """
    if isinstance(transformation, Identity):
        return True
    
    # For an affine transformation, we check if it is only scaling and translation
    # Given an affine 2D transform, checking that the off-diagonal values = 0 is sufficient to confirm it consists only of axis-aligned scaling and translation.
    if transformation.matrix[0,1] == 0 and transformation.matrix[1,0] == 0:
        return True
    
    return False

def _images_merscope(
    images_dir: str | Path,
    stainings: str | list[str] = "DAPI",
    z_layers: int | list[int] | None = None,
    aggregate_z: Literal["max", "mean"] | Callable | None = None,
    agg_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> list[DataTree]:
    """Retrieve and process images from a merscope output directory, optionally aggregating 
    across z-layers.

    Parameters
    ----------
    images_dir : str or Path
        The directory containing the images to be processed.
    stainings : str or list of str, optional
        The staining types to be used. Defaults to "DAPI".
    z_layers : int, list of int or None, optional
        The z-layers to be processed. If None, it assumes all z-layers will be used.
    aggregate_z : {'max', 'mean'} or Callable or None, optional
        The aggregation method to apply across z-layers.
    agg_kwargs : Mapping[str, Any], optional
        Additional keyword arguments for the aggregation function.
    image_models_kwargs : Mapping[str, Any], optional
        Keyword arguments for image model configuration.
    imread_kwargs : Mapping[str, Any], optional
        Additional keyword arguments for the image reading function.

    Returns
    -------
    list of DataTree
        A list containing the processed images, organized by dataset ID and z-layer.

    Raises
    ------
    KeyError
        If both z_layers and aggregate_z are None or if aggregate_z is not a recognized method.
    """
    from spatialdata_io.readers.merscope import _get_reader
    import glob
    import re
    
    if not isinstance(images_dir, Path):
        images_dir = Path(images_dir)

    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        image_models_kwargs["chunks"] = (1, 4096, 4096)

    if "scale_factors" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    images = {}

    if isinstance(z_layers, int):
        z_layers = [z_layers]
    elif z_layers is None:
        image_files = glob.glob(images_dir / 'mosaic_*_z*.tif')
        z_layers = set()
        for file in image_files:
            m = re.match(r'mosaic_(\S+)_z(\d+).tif', Path(file).name)
            if m is None:
                continue
            z_layers.add(int(m.groups()[1]))
        z_layers = sorted(list(z_layers))
        
    reader = _get_reader(None)

    dataset_id = f"{images_dir.parent.parent.name}_{images_dir.parent.name}"
    
    if aggregate_z:
        im = concat(
            [
                reader(images_dir,
                       [stainings] if isinstance(stainings, str) else stainings or [],
                       z_layer,
                       image_models_kwargs,
                       **imread_kwargs,
                )["scale0"].ds.to_dataarray()
                for z_layer in z_layers
            ],
            dim="z",
        )

        if callable(aggregate_z):
            im = im.reduce(aggregate_z, **agg_kwargs, dim="z")
        elif aggregate_z == "max":
            im = im.max(dim="z")
        elif aggregate_z == "mean":
            im = im.mean(dim="z")
        else:
            raise KeyError(
                f"{aggregate_z} not implemented. Try writing as callable function."
            )
        im = im.squeeze()
        parsed_image = Image2DModel.parse(im if "c" in im.dims else im.expand_dims("c", axis=0),
                                          c_coords=stainings,
                                          rgb=None,
                                          **image_models_kwargs)
        
        label = aggregate_z if isinstance(aggregate_z, str) else aggregate_z.__name__
        images[f"{dataset_id}_{label}_z_layers"] = parsed_image
    else:
        for z_layer in z_layers:
            images[f"{dataset_id}_z{z_layer}"] = reader(images_dir,
                                                        [stainings] if isinstance(stainings, str) else stainings or [],
                                                        z_layer,
                                                        image_models_kwargs,
                                                        **imread_kwargs)
            

    return images


def _images_xenium(
    xenium_dir: str | Path,
    stainings: str | list[str] = "DAPI",
    aggregate_z: Literal["max", "mean"] | Callable | None = None,
    agg_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> list[DataTree]:
    if not isinstance(xenium_dir, Path):
        xenium_dir = Path(xenium_dir)
    
    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        image_models_kwargs["chunks"] = (1, 1, 4096, 4096)
    
    images = {}
    im = da.expand_dims(imread(xenium_dir / 'morphology.ome.tif', **imread_kwargs), axis=0).rechunk((1, 1, 4096, 4096))
    im = Image3DModel.parse(im,
                            dims=('c', 'z', 'y', 'x'),
                            transformations={"global": Identity()},
                            c_coords=None,
                            scale_factors=None, # We don't want scale factors as they scale the z axis as well. Alternative could be messing with multiscale_spatial_image.to_multiscale
                           )
    
    if aggregate_z:
        if callable(aggregate_z):
            im = im.reduce(aggregate_z, **agg_kwargs, dim="z")
        elif aggregate_z == "max":
            im = im.max(dim="z")
        elif aggregate_z == "mean":
            im = im.mean(dim="z")
        else:
            raise KeyError(
                f"{aggregate_z} not implemented. Try writing as callable function."
            )
            
        parsed_image = Image2DModel.parse(im,
                                          c_coords=stainings,
                                          rgb=None,
                                          **image_models_kwargs)
        
        label = aggregate_z if isinstance(aggregate_z, str) else aggregate_z.__name__
        images[f"morphology_{label}_z_layers"] = parsed_image
    else:
        images[f"morphology"] = im
            
    return images


def _polygons(features, transformations, z_plane=None):
    """
    Custom function for loading polygons generated from spots-in-space for appending into a SpatialData

    Parameters:
    features : {geojson.FeatureCollection}
        sis formatted featurecollection. will be formatted to standard
    transformations : {spatialdata.transformations.BaseTransformation}
        Image transformation for polygon to pixel conversion
    z_plane : {int, list of ints, or None}, optional
        z_plane or planes that we select. If none, will take the union of polygons across all z_layers. if -1, use all z-planes

    Returns:
    geo_df : GeoDataFrame
        geopandas dataframe of polygons to add to SpatailData object.
    """
    import warnings
    
    # Find out if z-planes are in the geojson
    z_planes_present = "z_plane" in features['features'][0]

    # In previous versions of SIS, we had z_plane in the feature directly, not in properties. So we are maintaining backwards compatibility here.
    if 'z_plane' in features['features'][0]:
        features = geojson.FeatureCollection([{"geometry": feature["geometry"],
                                               "id": feature["id"],
                                               "properties": {'z_plane': feature['z_plane'], **feature['properties']},
                                              } for feature in features["features"]])

    # id must be in properties for geopandas from_features to read it
    features = geojson.FeatureCollection([{"geometry": feature["geometry"],
                                           "properties": {"id": feature["id"], **feature['properties']},
                                          } for feature in features['features']])
    
    # Generate geopandas df
    geo_df = gpd.GeoDataFrame.from_features(features)
    
    if z_planes_present:
        geo_df = geo_df[[x is not None for x in geo_df["z_plane"]]]
        geo_df.loc[:, "z_plane"] = geo_df["z_plane"].astype(float).astype(int)
        if z_plane != -1: # -1 means we take all z-planes
            # Select a layer or take the union across all layers
            if z_plane is None:
                geo_df = geo_df.dissolve(by="id").reset_index()
            else:
                if not isinstance(z_plane, list):
                    z_plane = [z_plane]
                geo_df = geo_df[geo_df["z_plane"].isin(z_plane)]

    # Remove empty or non-valid geometries
    valid_mask = (geo_df["geometry"].is_valid) & (~geo_df["geometry"].is_empty)
    if 100 * np.count_nonzero(valid_mask) / len(geo_df) > 5:
        warnings.warn(f"{np.count_nonzero(valid_mask)} ({100 * np.count_nonzero(valid_mask) / len(geo_df):.1f}% of total) invalid or empty geometries found and removed.")
    geo_df = geo_df[(geo_df["geometry"].is_valid) & (~geo_df["geometry"].is_empty)]

    # Combine Multipolygons by taking their convex hull. This might make for larger areas, but this usually represents a minority of cells.
    multi_mask = [type(x) is MultiPolygon for x in geo_df["geometry"]]
    if 100 * np.count_nonzero(multi_mask) / len(geo_df) > 5:
        warnings.warn(f"{np.count_nonzero(multi_mask)} ({100 * np.count_nonzero(multi_mask) / len(geo_df):.1f}% of total) multipolygons found. These will be converted to single polygons by taking their convex hulls.")
    geo_df.loc[:, "geometry"] = geo_df["geometry"].apply(
        lambda x: x.convex_hull if type(x) is MultiPolygon else x
    )
    
    return ShapesModel.parse(geo_df, transformations=transformations)


def _spottable(
    df: dd.DataFrame,
    transformations: BaseTransformation,
    z_planes: int | list[int] | None = None,
    genes: str | list[str] | None = None,
) -> dd.DataFrame:
    """
    Custom import function for appending spots-in-space spot-table files into spatialdata.

    Parameters:
    df: {dask.dataframe.core.DataFrame}
        Dask dataframe containing the transcript information
    transformations : spatialdata.transformations.BaseTransformation
        Image transformation for micron to pixel conversion.
    z_planes : {int, list of ints, or None}
        z_planes to include in the segmentation.
    genes : {str, list of str, or None}
        genes to inlcude in the spottable.

    Returns:
    transcripts : dask.dataframe.DataFrame
        dask.dataframe.DataFrame of transcript locations.
    """
    if z_planes:
        z_planes = [z_planes] if isinstance(z_planes, int) else z_planes
        df = df[df["z"].isin(z_planes)]

    if genes:
        genes = [genes] if isinstance(genes, str) else genes
        df = df[df["gene_names"].isin(genes)]

    transcripts = PointsModel.parse(
        df,
        coordinates={"x": "x", "y": "y", "z": "z"},
        transformations=transformations,
        feature_key="gene_names",
    )

    transcripts["gene"] = transcripts["gene_names"].astype("category")
    return transcripts


def _cell_by_gene(
    adata: anndata.AnnData,
) -> anndata.AnnData:
    try:
        del adata.uns['cell_polygons']
    except KeyError:
        pass
    return TableModel.parse(adata)


def merscope_to_spatialdata(
    images_dir: str | Path,
    sis_dir: str | Path | None = None,
    spot_table: SpotTable | None = None,
    z_layers: int | list[int] | None = list(range(7)),
    aggregate_z: Literal["max", "mean"] | Callable | None = None,
    agg_kwargs: Mapping[str, Any] = MappingProxyType({}),
    stainings: str | list[str] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    genes: str | list[str] | None = None,
    get_images: bool = True,
    get_cell_boundaries: bool = True,
    get_transcripts: bool = True,
    get_anndata: bool = True,
) -> SpatialData:
    """
    Parameters:

    image_dir: {str pathlib.Path}
        path to directory that contains images
    sis_dir : {str pathlib.Path}
        path to directory that contains sis segmentation results
    spot_table : {sis.SpotTable}
        sis.SpotTable that contains sis segmentation results
    z_layers : {int, list of int, or None}
        z_planes to include from merscope image. Must be None if aggregate_z is not None.
    aggregate_z : {"max", "mean", Callable, or None}
        function to aggregate_z all z_planes of a merscope image. Must be none if z_layers is not None.
    agg_kwargs : dict, optional
        kwargs to aggregate_z if aggregate_z is a callable function.
    stainings : {str or list of str}, optional
        stainings to get images for. Can only be [DAPI, PolyT, or Aux 1 - 3]. Will default to all stains in images folder.
    imread_kwargs : dict, optional
        kwargs to pass to _rioxarray_load_merscope function.
    image_models_kwargs : dict, optional
        kwargs to pass to Image2DModel.parse function.
    genes : {str, list of str, or None}, optional
        genes to inlcude in the spottable. Defaults to all genes.
    get_images : bool
        whether or not to load spatialdata with images. Defualt, True.
    get_cell_boundaries : bool
        whether or not to load sis segmentation polygons. Defualt, True.
    get_transcripts : bool
        whether or not to load spot_table. Defualt, True.
    get_anndata : bool
        whether or not load AnnData Object. Defualt, True.
    
    Returns:

    SpatialData of the the MERSCOPE images and the transcript locations.
    """
    if (sis_dir is None) == (spot_table is None):
        raise ValueError('One and exactly one of sis_dir and spot_table should be defined')

    # get images
    if isinstance(images_dir, str):
        images_dir = Path(images_dir)

    if get_images:
        images = _images_merscope(
            images_dir,
            stainings if stainings else _get_channel_names(images_dir),
            z_layers,
            aggregate_z,
            agg_kwargs,
            image_models_kwargs,
            imread_kwargs,
        )

    # get transfromations
    transform = Affine(
        np.genfromtxt(images_dir / "micron_to_mosaic_pixel_transform.csv"),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )

    transformations = {"global": transform}

    # get polygons
    if isinstance(sis_dir, str):
        sis_dir = Path(sis_dir)

    if get_cell_boundaries:
        if sis_dir is not None:
            with open(sis_dir / 'cell_polygons.geojson') as f:
                features = geojson.load(f)
        else:
             features = spot_table.get_geojson_collection()  
        shapes = {
            "sis_cell_polygons": _polygons(
                features, transformations, z_layers
            )
        }

    # get spottable
    if get_transcripts:
        if sis_dir is not None:
            df = dd.read_csv(sis_dir / "segmented_spot_table.csv", dtype={"cell_labels": str, "production_cell_ids": str})
        else:
            df = spot_table.dataframe(cols=['x', 'y', 'z', 'gene_ids', 'gene_names', 'cell_ids'])
            if spot_table.cell_labels is not None: 
                df['cell_labels'] = spot_table.cell_labels
            df = dd.from_pandas(df, npartitions=1).repartition(partition_size='64MB')
        points = {"segmented_spot_table": _spottable(df, transformations, z_layers, genes)}

    # get anndata
    if get_anndata:
        if sis_dir is not None:
            adata = anndata.read_h5ad(sis_dir / "cell_by_gene.h5ad")
        else:
            adata = spot_table.cell_by_gene_anndata('sparse')
        tables = {"cell_by_gene": _cell_by_gene(adata)}

    return SpatialData(images=images, points=points, shapes=shapes, tables=tables)


def xenium_to_spatialdata(
    xenium_dir: str | Path,
    sis_dir: str | Path | None = None,
    spot_table: SpotTable | None = None,
    aggregate_z: Literal["max", "mean"] | Callable | None = None,
    agg_kwargs: Mapping[str, Any] = MappingProxyType({}),
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    genes: str | list[str] | None = None,
    get_images: bool = True,
    get_cell_boundaries: bool = True,
    get_transcripts: bool = True,
    get_anndata: bool = True,
) -> SpatialData:
    """
    Parameters:

    xenium_dir: {str pathlib.Path}
        path to directory that contains images
    sis_dir : {str pathlib.Path}
        path to directory that contains sis segmentation results
    spot_table : {sis.SpotTable}
        sis.SpotTable that contains sis segmentation results
    aggregate_z : {"max", "mean", Callable, or None}
        function to aggregate_z all z_planes of a merscope image. Must be none if z_layers is not None.
    agg_kwargs : dict, optional
        kwargs to aggregate_z if aggregate_z is a callable function.
    imread_kwargs : dict, optional
        kwargs to pass to _rioxarray_load_merscope function.
    image_models_kwargs : dict, optional
        kwargs to pass to Image2DModel.parse function.
    genes : {str, list of str, or None}, optional
        genes to inlcude in the spottable. Defaults to all genes.
    get_images : bool
        whether or not to load spatialdata with images. Defualt, True.
    get_cell_boundaries : bool
        whether or not to load sis segmentation polygons. Defualt, True.
    get_transcripts : bool
        whether or not to load spot_table. Defualt, True.
    get_anndata : bool
        whether or not load AnnData Object. Defualt, True.
    
    Returns:

    SpatialData of the the MERSCOPE images and the transcript locations.
    """
    if (sis_dir is None) == (spot_table is None):
        raise ValueError('One and exactly one of sis_dir and spot_table should be defined')

    # get images
    if isinstance(xenium_dir, str):
        xenium_dir = Path(xenium_dir)

    if get_images:
        images = _images_xenium(
            xenium_dir,
            ['DAPI'],
            aggregate_z,
            agg_kwargs,
            image_models_kwargs,
            imread_kwargs,
        )

    # get transfromations
    tiff_image_file = tifffile.TiffFile(xenium_dir / 'morphology.ome.tif')

    # this file should have OME metadata:
    metadata_root = ET.fromstring(tiff_image_file.ome_metadata)
    # extract the pixel size 
    for child in metadata_root:
        if "Image" in child.tag:
            for cc in child:
                if "Pixels" in cc.tag:
                    um_per_pixel_x = float(cc.attrib["PhysicalSizeX"])
                    um_per_pixel_y = float(cc.attrib["PhysicalSizeY"])
    # and turn this into a transformation matrix
    affine_matrix = np.eye(3)[:2,:]
    affine_matrix[0,0] = um_per_pixel_x
    affine_matrix[1,1] = um_per_pixel_y
    m3 = np.eye(3)
    m3[:2] = affine_matrix
    um_to_pixel_matrix = np.linalg.inv(m3)
    
    transform = Affine(
        um_to_pixel_matrix,
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )

    transformations = {"global": transform}

    # get polygons
    if isinstance(sis_dir, str):
        sis_dir = Path(sis_dir)

    if get_cell_boundaries:
        if sis_dir is not None:
            with open(sis_dir / 'cell_polygons.geojson') as f:
                features = geojson.load(f)
        else:
             features = spot_table.get_geojson_collection()  
        shapes = {
            "sis_cell_polygons": _polygons(
                features, transformations, -1
            )
        }

    # get spottable
    if get_transcripts:
        if sis_dir is not None:
            df = dd.read_csv(sis_dir / "segmented_spot_table.csv", dtype={"cell_labels": str, "production_cell_ids": str})
        else:
            df = spot_table.dataframe(cols=['x', 'y', 'z', 'gene_ids', 'gene_names', 'cell_ids'])
            if spot_table.cell_labels is not None: 
                df['cell_labels'] = spot_table.cell_labels
            df = dd.from_pandas(df, npartitions=1).repartition(partition_size='64MB')
        points = {"segmented_spot_table": _spottable(df, transformations, genes)}

    # get anndata
    if get_anndata:
        if sis_dir is not None:
            adata = anndata.read_h5ad(sis_dir / "cell_by_gene.h5ad")
        else:
            adata = spot_table.cell_by_gene_anndata('sparse')
        tables = {"cell_by_gene": _cell_by_gene(adata)}

    return SpatialData(images=images, points=points, shapes=shapes, tables=tables)
