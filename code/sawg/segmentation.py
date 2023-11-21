from __future__ import annotations
import os, tempfile, pickle, traceback
import scipy.ndimage, scipy.interpolate
import numpy as np
from tqdm.notebook import tqdm
from .hpc import run_slurm_func
from .spot_table import SpotTable
from .image import Image, ImageBase, ImageTransform

from pathlib import Path
import datetime
import glob
import time
from abc import abstractmethod
from typing import Union
import pandas as pd
import anndata as ad
# used for type hint but it causes a circular import error
# from sawg.spatial_dataset import SpatialDataset


def run_segmentation(load_func, load_args:dict, subregion:dict|None, method_class, method_args:dict, result_file:str|None, cell_id_file:str|None):
    """Load a spot table, run segmentation (possibly on a subregion), and save the SegmentationResult.
    """
    spot_table = load_func(**load_args)
    print(f"loaded spot table {len(spot_table)}")
    if subregion is not None:
        spot_table = spot_table.get_subregion(*subregion)
    print(f"subregion {subregion} {len(spot_table)}")
    seg = method_class(**method_args)
    result = seg.run(spot_table)
    print(f"cell_ids {len(result.cell_ids)}")

    if result_file is not None:
        result.save(result_file)
        print(f"saved segmentation result to {result_file}")
    if cell_id_file is not None:
        np.save(cell_id_file, result.cell_ids)
        print(f"saved segmentated cell IDs to {cell_id_file}")


class SegmentationResult:
    """Represents a segmentation of SpotTable data--method, options, results
    """
    def __init__(self, method:'SegmentationMethod', input_spot_table:SpotTable):
        self.method = method
        self.input_spot_table = input_spot_table

    @property
    def cell_ids(self):
        """Array of segmented cell IDs for each spot in the table
        """
        raise NotImplementedError()        

    def spot_table(self, min_spots=None):
        """Return a new SpotTable with cell_ids determined by the segmentation.

        if min_spots is given, then it specifies the threshold below which cells will be discarded
        """
        cell_ids = self.cell_ids

        if min_spots is not None:
            cell_ids = cell_ids.copy()
            cids, counts = np.unique(cell_ids, return_counts=True)
            for cid, count in zip(cids, counts):
                if count < min_spots:
                    mask = cell_ids == cid
                    cell_ids[mask] = 0

        return self.input_spot_table.copy(cell_ids=cell_ids)

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))


class SegmentationMethod:
    """Base class defining segmentation methods.

    Subclasses should initialize with a dictionary of options, then calling
    run(spot_table) will execute the segmentation method and return a SegmentationResult.
    """
    def __init__(self, options:dict):
        self.options = options
        
    def run(self, spot_table:SpotTable):
        """Run segmentation on spot_table and return a Segmentation object.
        """
        raise NotImplementedError()
        
    def _get_spot_table(self, spot_table:SpotTable):
        """Return the SpotTable instance to run segmentation on. 
        
        If spot_table is a string, load from npz file.
        
        If a sub-region is specified in options, return the sub-table.
        """
        if isinstance(spot_table, str):
            spot_table = SpotTable.load_npz(spot_table)
           
        region = self.options.get('region', None)
        if region is not None:
            spot_table = spot_table.get_subregion(region)
            
        return spot_table
        


class CellposeSegmentationMethod(SegmentationMethod):
    """Implements 2D or 3D cellpose segmentation on SpotTable

    Will automatically segment from images attached to the SpotTable or
    generate an image from total mRNA.
    
    options = {
        'region': ((xmin, xmax), (ymin, ymax)),  # or None for whole table
        'cellpose_model': 'cyto2',
        'images': {
            'nuclei': 'DAPI',
            'cyto': 'total_mrna',
        },
        'px_size': 0.108,          # um / px
        'cell_dia': 10,            # um
        'z_plane_thickness': 1.5,  # um
        'cellpose_options': {
            'gpu': True,
            'batch_size': 8,
        },
        'dilate': 0,  # um - dilate segmentation after cellpose has finished
    }
    """
    
    def __init__(self, options):
        super().__init__(options)
        
    def run(self, spot_table):
        import cellpose.models
        spot_table = self._get_spot_table(spot_table)
        
        # collect all cellpose options
        cp_opts = {
            'z_axis': 0,
            'channel_axis': 3,
            'batch_size': 8,   # more if memory allows
            'normalize': True,
            'tile': False,
            'diameter': None # if None, cellpose will estimate diameter or use from pretrained model file
        }
        cp_opts.update(self.options['cellpose_options'])
        cp_opts.setdefault('anisotropy', self.options['z_plane_thickness'] / self.options['px_size'])
        if self.options['cell_dia'] is not None:
            manual_diam = self.options['cell_dia'] / self.options['px_size']
            cp_opts.update({'diameter': manual_diam})
        
        # collect images
        images = {}
        if 'nuclei' in self.options['images']:
            img_spec = self.options['images']['nuclei']
            images['nuclei'] = self._read_image_spec(img_spec, spot_table, px_size=self.options['px_size'], images=images)
        if 'cyto' in self.options['images']:
            img_spec = self.options['images']['cyto']
            img = self._read_image_spec(img_spec, spot_table, px_size=self.options['px_size'], images=images)
            images['cyto'] = img
        self.images = images

        # prepare image data for segmentation
        first_image = list(images.values())[0]
        cp_opts['do_3D'] = first_image.shape[0] > 1
        if len(images) == 2:
            assert images['cyto'].shape == images['nuclei'].shape
            cyto_data = images['cyto'].get_data()
            image_data = np.empty((images['cyto'].shape[:3]) + (3,), dtype=cyto_data.dtype)
            image_data[..., 0] = cyto_data
            image_data[..., 1] = images['nuclei'].get_data()
            image_data[..., 2] = 0
            channels = [1, 2]  # cyto=1 (red), nuclei=2 (green)
        else:
            image_data = first_image.get_data()
            channels = [0, 0]
            
        # decide whether to use GPU
        gpu = self.options.get('cellpose_gpu', False)
        self.gpu_msg = None
        if gpu == 'auto':
            try:
                import torch
                if torch.cuda.device_count() > 0:
                    gpu = True
                    self.gpu_msg = "enabled GPU"
                else:
                    gpu = False
                    self.gpu_msg = "no GPU found"
            except Exception as exc:
                gpu = False
                self.gpu_msg = ''.join(traceback.format_exc())

        # initialize cellpose model
        cp_opts['channels'] = channels

        if self.options['cellpose_model'] in ['cyto', 'cyto2', 'nuclei']:
            # use a default cellpose 1.0 model
            model = cellpose.models.Cellpose(model_type=self.options['cellpose_model'], gpu=gpu)
            # run segmentation
            masks, flows, styles, diams = model.eval(image_data, **cp_opts)
            cellpose_output = {
                'masks': masks, 
                'flows': flows, 
                'styles': styles, 
                'diams': diams,
            }

        else:
            # use a path to a custom model
            assert os.path.exists(self.options['cellpose_model'])
            model = cellpose.models.CellposeModel(pretrained_model=self.options['cellpose_model'], gpu=gpu)
            # run segmentation
            masks, flows, styles = model.eval(image_data, **cp_opts)
            cellpose_output = {
                'masks': masks, 
                'flows': flows, 
                'styles': styles, 
            }

        dilate = self.options.get('dilate', 0)
        if dilate != 0:
            masks = dilate_labels(masks, radius=dilate/self.options['px_size'])
            cellpose_output.update({'masks': masks})

        # return result object
        result = CellposeSegmentationResult(
            method=self, 
            input_spot_table=spot_table,
            cellpose_output=cellpose_output,
            image_transform=first_image.transform,
        )
            
        return result

    def _read_image_spec(self, img_spec, spot_table, px_size, images):
        """Return an Image to be used in segmentation based on img_spec:
        
        - An Image instance is returned as-is
        - "total_mrna" returns an image generated from spot density
        - Any other string returns an image channel attached to the spot table
        - {'channel': channel, 'frame': int} can be used to select a single frame
        - {'channel': 'total_mrna', 'n_planes': int, 'frame': int, 'gauss_kernel': (1, 3, 3), 'median_kernel': (2, 10, 10)} can be ued to configure total mrna image generation
        """
        # optionally, cyto image may be generated from spot table total mrna        
        if img_spec is None or isinstance(img_spec, ImageBase):
            return img_spec
        if isinstance(img_spec, str):
            img_spec = {'channel': img_spec}
        if not isinstance(img_spec, dict):
            raise TypeError(f"Bad image spec: {img_spec}")

        if img_spec['channel'] == 'total_mrna':
            opts = img_spec.copy()
            opts.pop('channel')
            opts.update(self._suggest_image_spec(spot_table, px_size, images))
            return self.get_total_mrna_image(spot_table, **opts)
        else:
            return spot_table.get_image(channel=img_spec['channel'], frame=img_spec.get('frame', None))

    def _suggest_image_spec(self, spot_table, px_size, images):
        """Given a pixel size, return {'image_shape': shape, 'image_transform': tr} covering the entire area of spot_table.
        If any images are already present, use those as templates instead.
        """
        if len(images) > 0:
            img = list(images.values())[0]
            return {'image_shape': img.shape[:3], 'image_transform': img.transform}

        bounds = np.array(spot_table.bounds())
        scale = 1 / px_size
        shape = np.ceil((bounds[:, 1] - bounds[:, 0]) * scale).astype(int)
        
        tr_matrix = np.zeros((2, 3))
        tr_matrix[0, 0] = tr_matrix[1, 1] = scale
        tr_matrix[:, 2] = -scale * bounds[:, 0]
        image_tr = ImageTransform(tr_matrix)
        
        return {'image_shape': (1, shape[0], shape[1]), 'image_transform': image_tr}

    def get_total_mrna_image(self, spot_table, image_shape:tuple, image_transform:ImageTransform, n_planes:int, frame:int|None=None, gauss_kernel=(1, 3, 3), median_kernel=(2, 10, 10)):

        image_shape_full = (n_planes, *image_shape[1:3])
        density_img = np.zeros(image_shape_full, dtype='float32')

        spot_px = self.map_spots_to_img_px(spot_table, image_transform=image_transform, image_shape=image_shape_full)
        for i in range(n_planes):
            z_mask = spot_px[..., 0] == i
            x = spot_px[z_mask, 1]
            y = spot_px[z_mask, 2]
            bins = [
                # is this correct? test on rectangular tile
                np.arange(image_shape[1]+1),
                np.arange(image_shape[2]+1),
            ]
            density_img[i] = np.histogram2d(x, y, bins=bins)[0]

        import scipy.ndimage
        # very sensitive to these parameters :/
        density_img = scipy.ndimage.gaussian_filter(density_img, gauss_kernel)
        density_img = scipy.ndimage.median_filter(density_img, median_kernel)

        if frame is not None:
            return Image(density_img[..., np.newaxis], transform=image_transform, channels=['Total mRNA'], name=None).get_frame(frame)

        else:
            return Image(density_img[..., np.newaxis], transform=image_transform, channels=['Total mRNA'], name=None)

    def map_spots_to_img_px(self, spot_table:SpotTable, image:Image|None=None, image_transform:ImageTransform|None=None, image_shape:tuple|None=None):
        """Map spot table (x, y, z) positions to image (frame, row, col). 

        Optionally, provide the *image_transform* and *image_shape* instead of *image*.
        """
        if image is not None:
            assert image_transform is None and image_shape is None
            image_shape = image.shape
            image_transform = image.transform

        spot_xy = spot_table.pos[:, :2]
        spot_px_rc = np.floor(image_transform.map_to_pixels(spot_xy)).astype(int)

        # for this dataset, z values are already integer index instead of um
        if spot_table.z is None:
            spot_px_z = np.zeros(len(spot_table), dtype=int)
        else:
            spot_px_z = spot_table.z.astype(int)

        # some spots may be a little past the edge of the image; 
        # just clip these as they'll be discarded when tiles are merged anyway
        spot_px_rc[:, 0] = np.clip(spot_px_rc[:, 0], 0, image_shape[1]-1)
        spot_px_rc[:, 1] = np.clip(spot_px_rc[:, 1], 0, image_shape[2]-1)
        
        return np.hstack([spot_px_z[:, np.newaxis], spot_px_rc])


class CellposeSegmentationResult(SegmentationResult):
    def __init__(self, method:SegmentationMethod, input_spot_table:SpotTable, cellpose_output:dict, image_transform:ImageTransform):
        super().__init__(method, input_spot_table)
        self.cellpose_output = cellpose_output
        self.image_transform = image_transform
        self._cell_ids = None
        
    @property
    def cell_ids(self):
        """Array of segmented cell IDs for each spot in the table
        """
        # use segmented masks to assign each spot to a cell
        if self._cell_ids is None:
            spot_table = self.input_spot_table
            mask_img = self.mask_image
            spot_px = self.method.map_spots_to_img_px(spot_table, mask_img)

            # assign segmented cell IDs to a new table
            masks = mask_img.get_data()
            if self.cellpose_output['masks'].ndim == 2:
                # 2D mask
                self._cell_ids = masks[0, spot_px[:,1], spot_px[:,2]]
            else:
                # 3D mask
                self._cell_ids = masks[spot_px[:,0], spot_px[:,1], spot_px[:,2]]
        
        return self._cell_ids

    @property
    def mask_image(self):
        """Return Image instance containing mask data and pixel transform
        """        
        # annotate segmentation mask with spot-to-pixel transform
        masks = self.cellpose_output['masks']
        if masks.ndim == 2:
            # add frame and channel axes back
            masks = masks[np.newaxis, ..., np.newaxis]
        else:
            # add channel axis back
            masks = masks[..., np.newaxis]
        return Image(data=masks, transform=self.image_transform, channels=['Cellpose Mask'], name=None)
        
    def show(self, ax):
        """Display segmentation result in matplotlib axes
        """
        

class BaysorSegmentationMethod(SegmentationMethod):
    """Implements 3D Baysor segmentation on SpotTable

    Requires a Baysor binary to be executable on the system.

    options = {
        'region': ((xmin, xmax), (ymin, ymax)),  # or None for whole table
        'baysor_bin': '/path/to/Baysor',
        'baysor_output_path': '/path/to/baysor/data',
        'use_prior_segmentation': True,
        'no_gene_names': False,          # if true, remove gene names
        'cell_dia': 10,                  # um
        'z_plane_thickness': 1.5,        # um
        'baysor_options': {
            'scale-std': '25%',
            'prior-segmentation-confidence': None,
            'n-clusters': None,
            'no-ncv-estimation': True,
        }
    }
    """

    def __init__(self, options):
        super().__init__(options)

    def run(self, spot_table):
        spot_table = self._get_spot_table(spot_table)

        # collect all cellpose options
        baysor_opts = {
            'scale': self.options['cell_dia'],
            'scale-std': '25%',
            'prior-segmentation-confidence': None,
            'n-clusters': None,
            'no-ncv-estimation': True,
        }
        baysor_opts.update(self.options['baysor_options'])

        # correct Z positions from layers to micrometers
        # Note: ideally we shouldn't need to do this -- the merscope data should just have correct z coordinates to begin with.
        baysor_spot_table = spot_table.copy(deep=True)
        baysor_spot_table.pos[:,2] *= self.options['z_plane_thickness']

        # remove gene IDs--only use spot position to perform segmentation
        if self.options['no_gene_names']:
            baysor_spot_table.gene_ids[:] = 0

        # using --num-cells should increase convergence speed
        if spot_table.cell_ids is not None:
            num_cells = len(np.unique(spot_table.cell_ids))
            baysor_opts.setdefault('num-cells-init', num_cells)
        else:
            num_cells = None

        save_columns = ['x', 'y', 'z', 'gene_id']
        if self.options['use_prior_segmentation']:
            save_columns.append('cell_id')

        # we communicate with baysor by writing files to disk, so pick a place to do that
        output_path = self.options.get('baysor_output_path', None)
        out_is_tmp = False
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix='baysor_run_')
            out_is_tmp = True

        inp_file = os.path.join(output_path, 'input_spot_table.csv')
        out_file = os.path.join(output_path, 'output_spot_table.csv')

        # save csv to tmp
        baysor_spot_table.save_csv(inp_file, columns=save_columns)

        cmd = f"{self.options['baysor_bin']} run -o {out_file}"
        for arg,val in baysor_opts.items():
            if val is None:
                continue
            if val is True:
                cmd += f" --{arg}"
            else:
                cmd += f" --{arg} {val}"
        cmd += f" -x x -y y -z z -g gene_id {inp_file}"
        if self.options['use_prior_segmentation']:
            cmd += " :cell_id"
        self.baysor_command = cmd

        # run baysor
        os.system(cmd)

        # return result object
        result = BaysorSegmentationResult(
            method=self,
            input_spot_table=spot_table,
            baysor_command=cmd,
            baysor_output=out_file,
        )

        return result


def load_baysor_result(result_file, remove_noise=True, remove_no_cell=True, brl_output = False):
    if brl_output:
        dtype = [('x', 'float32'), ('y', 'float32'), ('z', 'float32'),('gene',str),('cluster', int), ('cell', int), ('is_noise', bool)]

        converters = {
            6: lambda x: x == 'true',
        }
        result_data = np.loadtxt(
            result_file,
            skiprows=1,
            usecols=[0, 1, 2, 3,4, 5, 6],
            delimiter=',',
            dtype=dtype,
            converters=converters
        )
    else:
        dtype = [('x', 'float32'), ('y', 'float32'), ('z', 'float32'),('cluster', int), ('cell', int), ('is_noise', bool)]

        converters = {
            9: lambda x: x == 'true',
        }
        result_data = np.loadtxt(
            result_file,
            skiprows=1,
            usecols=[0, 1, 2, 6, 7, 9],
            delimiter=',',
            dtype=dtype,
            converters=converters
        )

    z_vals = np.unique(result_data['z'])
    if remove_noise:
        result_data = result_data[~result_data['is_noise']]
    if remove_no_cell:
        result_data = result_data[result_data['cell'] > 0]

    return result_data


class BaysorSegmentationResult(SegmentationResult):
    def __init__(self, method:SegmentationMethod, input_spot_table:SpotTable, baysor_command:str, baysor_output:str):
        super().__init__(method, input_spot_table)
        self.baysor_command = baysor_command
        self.baysor_output = baysor_output
        self._cell_ids = None

    @property
    def cell_ids(self):
        """Array of segmented cell IDs for each spot in the table
        """
        # use segmented masks to assign each spot to a cell
        if self._cell_ids is None:
            spot_table = self.input_spot_table

            self.result = load_baysor_result(self.baysor_output, remove_noise=False, remove_no_cell=False)
            self._cell_ids = self.result['cell']

        return self._cell_ids



def dilate_labels(img, radius):
    """Dilate labeled regions of an image.

    Given an image with 0 in the background and objects labeled with different integer values (such
    as a cell segmentation mask), return a new image with objects expanded by *radius*.

    (Credit: https://stackoverflow.com/a/70261747)
    """
    mask = img > 0

    # fill in all pixels with nearest non-zero value
    inds = np.where(mask)
    interpolator = scipy.interpolate.NearestNDInterpolator(np.transpose(inds), img[inds])
    interpolated = interpolator(*np.indices(img.shape))

    # make a dilated mask to return background pixels to 0
    ri = int(np.ceil(radius))
    if img.ndim == 2:
        circle = np.fromfunction(lambda i,j: (((i-ri)**2 + (j-ri)**2) ** 0.5) < radius, (ri*2+1, ri*2+1)).astype(int)
    elif img.ndim == 3:
        circle = np.fromfunction(lambda i,j,k: (((i-ri)**2 + (j-ri)**2 + (k-ri)**2) ** 0.5) < radius, (ri*2+1, ri*2+1, ri*2+1)).astype(int)
    else:
        raise TypeError(f"Not implemented for {img.ndim}D images")
    dilated_mask = scipy.ndimage.grey_dilation(mask, footprint=circle)
    interpolated[~dilated_mask] = 0

    return interpolated


def get_segmentation_region(spot_table: SpotTable, subregion: tuple|str):
    """Get a subregion of a spot table to use for segmentation.

    Parameters
    ----------
    spot_table: SpotTable
        The spot table to subset.
    subregion: tuple|str
        Can be provided as a tuple of ((x_min, x_max), (y_min, y_max) to use as bounding box.
        Or provide a string corresponding to the name of an image channel attached to spot_table.
        If a string, will get the area of the spot table bounded by the image region.

    Returns
    -------
    subtable
        The subset spot table.
    """

    if isinstance(subregion, str):
        subrgn = spot_table.get_image(channel=subregion).bounds()

    else:
        subrgn = subregion

    subtable = spot_table.get_subregion(xlim=subrgn[0], ylim=subrgn[1])

    return subtable


def get_tiles(spot_table: SpotTable, max_tile_size: int = 200, overlap: int = 30):
    """Split a spot table into overlapping tiles.

    Returns a list of tiles and a list of corresponding subregions.
    """

    tiles = spot_table.grid_tiles(max_tile_size=max_tile_size, overlap=overlap)
    regions = [tile.parent_region for tile in tiles]

    return tiles, regions


def create_seg_run_spec(regions, load_func, load_args: dict, seg_method: SegmentationMethod, seg_opts: dict, tile_save_path: str):
    """Create run specifications for tiled segmentation on the hpc.

    Parameters
    ----------
    regions
        List of subregions corresponding to tiles of a spot table.
    load_func
        Function used to load a spatial dataset, e.g. SpotTable.load_merscope
    load_args: dict
        Keyword arguments to pass to the loading function.
    seg_method: SegmentationMethod
        The method to use for segmentation.
    seg_opts: dict
        Keyword arguments to pass to the segmentation method.
    tile_save_path: str
        Directory in which to save segmentation results and cell id files for individual tiles.

    Returns
    -------
    run_spec
        The specifications to segment each tile on the hpc.
    """

    assert os.path.exists(tile_save_path)

    run_spec = {}
    for i, region in enumerate(regions):
        run_spec[i] = (
            run_segmentation,
            (),
            dict(
                load_func=load_func,
                load_args=load_args,
                subregion=region,
                method_class=seg_method,
                method_args=seg_opts,
                result_file=os.path.join(f'{tile_save_path}/', f'segmentation_result_{i}.pkl'),
                cell_id_file=os.path.join(f'{tile_save_path}/', f'segmentation_result_{i}.npy'),
            )
        )
    print(f"Generated segmentation spec for {len(run_spec)} tiles")

    return run_spec


def run_segmentation_on_hpc(run_spec: dict, conda_env: str, hpc_host: str, job_path: str, **hpc_opts):
    """Submit an array job to run segmentation on tiles to hpc.
    Jobs are submitted immediately upon calling this function.

    Parameters
    ----------
    run_spec: dict
        The specifications to segment each tile on the hpc.
    conda_env: str
        Path to a conda environment from which to run segmentation.
    hpc_host: str
        Set as 'hpc-login' if running from a local machine,
        'localhost' if running from the hpc.
    job_path: str
        Directory in which to save job output and error log files.
    **hpc_opts
        Remaining keyword arguments to pass to run_slurm_func.

    Returns
    -------
    jobs
        A SlurmJob object with information about submitted jobs.
    """

    hpc_config = {
        'run_spec': run_spec,
        'conda_env': conda_env,
        'hpc_host': hpc_host,
        'job_path': job_path,
        'partition': 'celltypes',
        'job_name': 'segment',
        'nodes': 1,
        'ntasks': 1,
        'array': f'0-{len(run_spec)-1}',
        'mincpus': 1,
        'gpus_per_node': 1,
        'mem': '20G',
        'time': '0:30:00',
        'mail_user': None,
    }

    hpc_config.update(**hpc_opts)
    jobs = run_slurm_func(**hpc_config)

    return jobs


def merge_segmentation_results(spot_table: SpotTable, run_spec: dict, tiles: list[SpotTable]|None = None):
    """Merge results of a tiled segmentation, updating spot_table.cell_ids in place.

    Parameters
    ----------
    spot_table: SpotTable
        The entire spot table to which to assign cell_ids.
    run_spec: dict
        The specifications provided to segment each tile on the hpc.
    tiles: list[SpotTable]|None
        The individual tiles that were segmented.
        If not provided, will be generated from spot_table and run_spec.

    Returns
    ------- 
    cell_ids
        The cell_ids assigned to each spot in spot_table.
    merge_results
        Information about conflicts collected during tile merging.
    skipped
        Indices of tiles skipped during segmentation.
    """

    spot_table.cell_ids = np.empty(len(spot_table), dtype=int)
    spot_table.cell_ids[:] = -1

    merge_results = []
    skipped = []
    for i, tile_spec in enumerate(tqdm(run_spec.values())):
        cell_id_file = tile_spec[2]['cell_id_file']
        if not os.path.exists(cell_id_file):
            print(f"Skipping tile {i} : no cell ID file generated")
            skipped.append(i)
            continue

        if tiles is not None:
            # Use tiles in memory
            tile = tiles[i]

        else:
            # Recreate each tile from spot_table and run_spec
            tile_rgn = tile_spec[2]['subregion']
            tile = spot_table.get_subregion(xlim = tile_rgn[0], ylim = tile_rgn[1])

        tile.cell_ids = np.load(cell_id_file)
        result = spot_table.merge_cells(tile, padding=5)
        merge_results.append(result)

    cell_ids = spot_table.cell_ids

    return cell_ids, merge_results, skipped


class SegmentationRun:
    """Trying out an object to represent an individual segmentation run on a 
    dataset.
    """
    def __init__(
            self, 
            dt_file: Path|str,
            image_path: Path|str,
            output_dir: Path|str,
            dt_cache: Path|str|None,
            subrgn: str|tuple,
            seg_method: SegmentationMethod,
            seg_opts: dict,
            hpc_opts: dict
            ):

        # input/output paths
        self.images_path = image_path
        self.detected_transcripts_file = dt_file
        self.detected_transcripts_cache = dt_cache 
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # intermediate file paths
        self.regions_path = os.path.join(output_dir, 'regions.json')
        self.run_spec_path = os.path.join(output_dir, 'run_spec.pkl')
        self.tile_save_path = os.path.join(output_dir, 'seg_tiles/')
        self.cid_path = os.path.join(output_dir, 'segmentation.npy')
        self.cbg_path = os.path.join(output_dir, 'cell_by_gene.h5ad')

        # segmentation parameters
        self.subrgn = subrgn
        self.seg_method = seg_method
        self.seg_opts = seg_opts
        self.hpc_opts = hpc_opts

        # initial arguments saved for future reference 
        self.meta_path = os.path.join(output_dir, 'metadata.pkl')
        self.meta = {
                'dt_file': dt_file,
                'image_path': image_path,
                'output_dir': output_dir,
                'dt_cache': dt_cache,
                'subrgn': subrgn,
                'seg_method': seg_method,
                'seg_opts': seg_opts,
                'hpc_opts': hpc_opts
            }
        self.save_metadata()

        # load the spot table corresponding to the segmentation region
        self.spot_table = self.load_spot_table(subrgn)

    @abstractmethod
    def get_load_func(self):
        return

    @abstractmethod
    def get_load_args(self):
        return

    def save_metadata(self):
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)

    def load_metadata(self):
        with open(self.meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        return meta

    def save_regions(self, regions):
        regions_df = pd.DataFrame(regions, columns=['xlim', 'ylim'])
        regions_df.to_json(self.regions_path)

    def load_regions(self):
        assert os.path.exists(self.regions_path)
        regions = pd.read_json(self.regions_path).values
        regions = [tuple(r) for r in regions]
        return regions

    def save_run_spec(self, run_spec):
        with open(self.run_spec_path, 'wb') as f:
            pickle.dump(run_spec, f)

    def load_run_spec(self):
        with open(self.run_spec_path, 'rb') as f:
            run_spec = pickle.load(f)

        return run_spec

    def save_cell_ids(self, cell_ids):
        np.save(self.cid_path, cell_ids)

    def load_cell_ids(self):
        assert os.path.exists(self.cid_path)
        cell_ids = np.load(self.cid_path)
        return cell_ids

    def save_cbg(self, cell_by_gene):
        cell_by_gene.write(self.cbg_path)

    def load_cbg(self):
        assert os.path.exists(self.cbg_path)
        cell_by_gene = ad.read_h5ad(self.cbg_path)
        return cell_by_gene

    def load_spot_table(self, subregion):
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        subtable = get_segmentation_region(table, subregion)
        return subtable

    def check_run_status(self):
        """Check which steps are complete."""
        if os.path.exists(self.tile_save_path):
            num_segmented_tiles = len(os.listdir(self.tile_save_path))
        else:
            num_segmented_tiles = 0

        steps = {
                'tiled': os.path.exists(self.regions_path),
                'run spec': os.path.exists(self.run_spec_path),
                'jobs submitted': num_segmented_tiles > 0,
                'merged tiles': os.path.exists(self.cid_path),
                'cell by gene': os.path.exists(self.cbg_path)
        }

        return steps

    def do_pipeline(self):
        """Run steps as a pipeline.

        If the output dir does not exist, start a new segmentation.
        Otherwise, resume from an existing segmentation.
        If existing segmentation is complete: load and return the output files.
        """
        step_status = self.check_run_status()

        step_funcs = [
                self.tile_seg_region,
                self.get_seg_run_spec,
                self.submit_seg_jobs,
                self.merge_segmented_tiles,
                self.create_cell_by_gene
            ]
        
        not_done = [i for i, s in enumerate(list(step_status.values())) if not s] 
        steps_to_do = [step_funcs[i] for i in not_done]
        
        if len(steps_to_do) > 0:
            # evaluate steps not done
            [f() for f in steps_to_do]

        # Load and return cell by gene table
        cell_by_gene = self.load_cbg()

        return self.spot_table, cell_by_gene

    def track_job_progress(self, jobs, run_spec):
        """Track progress of segmentation jobs. Will not complete until all 
        cell id files have been saved.
        """
        saved_cid_files = [os.path.exists(v[2]['cell_id_file']) for v in run_spec.values()]
        timeout = time.time() + 60*30 # 30 min, the standard time for one job
        with tqdm(total=len(jobs)) as pbar:
            while not all(saved_cid_files):
                finished = [j for j in jobs if os.path.exists(j.output_file)]
                n_finished = len(finished)
                saved_cid_files = [os.path.exists(v[2]['cell_id_file']) for v in run_spec.values()]
                n_saved = saved_cid_files.count(True)
                pbar.update(n_saved - pbar.n)
                print(f'Submitted: {n_finished} / {len(jobs)} Status: {jobs.state_counts()} Saved: {n_saved}')
                if not any(saved_cid_files) and time.time() > timeout:
                    jobs.cancel()
                    raise RuntimeError('No cell id files found within the timeout period.'\
                            'Jobs canceled. Check error logs.')
                time.sleep(60)

    def tile_seg_region(self, **kwargs):
        print('Tiling segmentation region...')
        subtable = self.spot_table
        tiles, regions = get_tiles(subtable, **kwargs)
        
        # save regions
        self.save_regions(regions)

        return tiles, regions

    def get_seg_run_spec(self):
        regions = self.load_regions()
        seg_method = self.seg_method
        seg_opts = self.seg_opts
        if not os.path.exists(self.tile_save_path):
            os.mkdir(self.tile_save_path)

        run_spec = create_seg_run_spec(
                regions,
                self.get_load_func(),
                self.get_load_args(),
                seg_method,
                seg_opts,
                self.tile_save_path
        )

        # save run_spec
        self.save_run_spec(run_spec)

        return run_spec

    def submit_seg_jobs(self):
        run_spec = self.load_run_spec()
        hpc_opts = self.hpc_opts

        job_path = os.path.join(self.output_dir, 'hpc-jobs/')
        if not os.path.exists(job_path):
            os.mkdir(job_path)
        hpc_opts.update({'job_path': job_path})

        jobs = run_segmentation_on_hpc(run_spec, **hpc_opts)
        self.track_job_progress(jobs, run_spec)

        return jobs

    def merge_segmented_tiles(self):
        run_spec = self.load_run_spec()
        
        subtable = self.spot_table
        # need to make extra sure this updates the spot table attached to the 
        # class in place
        print('Merging tiles...')
        cell_ids, merge_results, skipped = merge_segmentation_results(subtable, run_spec, None)

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save cell_ids
        self.save_cell_ids(cell_ids)

        return cell_ids, merge_results, skipped

    def create_cell_by_gene(self, remove_bg = True):
		# to be updated with volume and other metadata calculation when done
        subtable = self.spot_table
        subtable = subtable.filter_cells(real_cells = remove_bg)

        cell_by_gene = subtable.cell_by_gene_anndata()
        self.save_cbg(cell_by_gene)

        return cell_by_gene


class MerscopeSegmentationRun(SegmentationRun):
    def __init__(
            self, 
            dt_file: Path|str, 
            image_path: Path|str, 
            output_dir: Path|str, 
            dt_cache: Path|str|None,
            subrgn: str|tuple,
            seg_method: SegmentationMethod,
            seg_opts: dict,
            hpc_opts: dict,
        ):
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, hpc_opts)

    @classmethod
    def from_spatial_dataset(cls, sp_dataset, output_dir, subrgn, seg_method, seg_opts, hpc_opts):
        """Alternate constructor to load from a SpatialDataset"""
        image_path = sp_dataset.images_path
        csv_file = sp_dataset.detected_transcripts_file
        cache_file = sp_dataset.detected_transcripts_cache

        return cls(csv_file, image_path, output_dir, cache_file, subrgn, seg_method, seg_opts, hpc_opts)

    @classmethod
    def from_expt_path(cls, expt_path: Path, cache_file: Path|str, output_dir: Path|str):
        image_path = expt_path / 'images/'
        csv_file = expt_path / 'detected_transcripts.csv'
        return cls(csv_file, image_path, cache_file, output_dir)

    @classmethod
    def from_timestamp(cls, output_dir, timestamp):
        """Load a partially completed run from timestamp (metadata?)"""
        path_to_meta = os.path.join(output_dir, timestamp, 'metadata.pkl')

        with open(path_to_meta, 'rb') as f:
            meta = pickle.load(f)

        return cls(**meta['init_args'])

    def get_load_func(self):
        """Get the function to load a spot table."""
        return SpotTable.load_merscope

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
        load_args = {
                'image_path': self.images_path,
                'csv_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
                'max_rows': None
        }
        return load_args

class StereoSeqSegmentationRun(SegmentationRun):
    def __init__(
            self, 
            dt_file: Path|str, 
            image_path: Path|str, 
            output_dir: Path|str, 
            dt_cache: Path|str|None,
            subrgn: str|tuple,
            seg_method: SegmentationMethod,
            seg_opts: dict,
            hpc_opts: dict,
        ):
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, hpc_opts)

    def get_load_func(self):
        """Get the function to load a spot table."""
        return SpotTable.load_stereoseq

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
        load_args = {
                'image_path': self.image_path,
                'gem_file': self.dt_file,
                'cache_file': self.dt_cache,
                'skiprows': 7,
                'image_channel': 'nuclear'
        }
        return load_args
