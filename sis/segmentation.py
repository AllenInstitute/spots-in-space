from __future__ import annotations
import os, tempfile, pickle, traceback
import scipy.ndimage, scipy.interpolate
import numpy as np
from tqdm.autonotebook import tqdm
from .hpc import SlurmJob, SlurmJobArray, run_slurm_func, double_mem, memory_to_bytes, slurm_time_to_seconds, seconds_to_time
from .spot_table import SpotTable, SegmentedSpotTable
from .image import Image, ImageBase, ImageTransform

import inspect
import json
from pathlib import Path, PurePath
import time
from abc import abstractmethod
import pandas as pd
import anndata as ad
import sis
from sis.util import convert_value_nested_dict

from .optional_import import optional_import
geojson = optional_import('geojson')

def run_segmentation(load_func, load_args:dict, subregion:dict|None, method_class, method_args:dict, result_file:str|None, cell_id_file:str|None):
    """Load a spot table, run segmentation (possibly on a subregion), and save the SegmentationResult.

    Parameters
    ----------
    load_func :
        The method of SpotTable used to load a dataset (e.g. SpotTable.load_merscope).
    load_args : dict
        Parameters passed to load_func.
    subregion : dict, optional
        The subregion of the SpotTable to segment.
    method_class :
        The SegmentationMethod used for segmentation.
    method_args : dict
        The arguments to pass to method_class.
    result_file : str, optional
        Where to save the SegmentationResult object.
    cell_id_file : str, optional
        Where to save the list of cell ids output by segmentation.
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
        """Return a SegmentedSpotTable with cell_ids determined by the segmentation.

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

        return SegmentedSpotTable(self.input_spot_table, cell_ids)

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
        """Create a total mRNA image (histogram of spot density) from the spot table.
        Can be used to approximate cytosol staining for segmentation.
        Smoothing can optionally be applied.

        Parameters
        ----------
        spot_table : SpotTable
            The spot table used to create the image.
        image_shape : tuple
            The shape of the image.
        image_transform : ImageTransform
            The transform that relates image and spot coordinates (?).
        n_planes : int
            The number of z planes in the image.
        frame : int, optional
            The frame (specific z plane) used to create the image, if wanting to create a 2D image from a 3D image.
        gauss_kernel : tuple
            Kernel used for gaussian smoothing of the image. Default (1, 3, 3).
        median_kernel : tuple
            Kernel used for median smoothing of the image. Default (2, 10, 20).
        """

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



class SegmentationPipeline:
    """Base class for running segmentation on a whole section (or subregion).
    When the class is initialized, it creates the segmentation output directory
    and sets paths for intermediate files. Each step has defined input and 
    output files. Steps can be run individually by calling their respective 
    methods or in a defined sequence by calling the run() method.

    Currently there is an assumption that the attributes set upon initialization
    are not changed by the user. If you change them, be warned that the 
    metadata dictionary and spot table subregion may become outdated if the 
    appropriate methods are not called afterward.

    Parameters
    ----------
    dt_file: str or Path
        Path to the detected transcripts file.
    image_path: str or Path
        Path to the images.
    output_dir: str or Path
        Where to save output files.
    dt_cache: str or Path, optional
        Path to the detected transcripts cache file. Used for faster loading.
    subrgn: str or tuple
        The subregion to segment. Set to a string, e.g. 'DAPI', to segment the 
        full region bounded by the associated image channel. To segment a 
        smaller region, set to a tuple corresponding to a bounding box.
    seg_method: SegmentationMethod
        The segmentation method to use. Must be found in sis.segmentation.
    seg_opts: dict
        Options to pass to seg_method.
    polygon_opts: dict, optional
        Options to pass to for cell polygon generation. Currently supports save_file_extension and alpha_inv_coeff.
        Default is None, which sets save_file_extension to 'geojson' and alpha_inv_coeff to 4/3.    
    seg_hpc_opts: dict, None, optional
        Options to use for segmenting tiles on the hpc. Default is None
    polygon_hpc_opts: dict, None, optional
        Options to use for calculating cell polygons on the hpc. Default is None
    hpc_opts: dict, None, optional
        Options to use for both segmenting tiles and calculating cell polygons on the hpc (can be used in place of submitting both seg_hpc_opts and polygon_hpc_opts).
        Default is None
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
            polygon_opts: dict|None=None,
            seg_hpc_opts: dict|None=None,
            polygon_hpc_opts: dict|None=None,
            hpc_opts: dict|None=None
            ):

        # input/output paths
        self.image_path = image_path
        self.detected_transcripts_file = dt_file
        self.detected_transcripts_cache = dt_cache 

        if isinstance(output_dir, str) or isinstance(output_dir, PurePath):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        if (seg_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either seg_hpc_opts or hpc_opts must be provided.")
        if (polygon_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either polygon_hpc_opts or hpc_opts must be provided.")

        # segmentation parameters
        self.subrgn = subrgn
        self.seg_method = seg_method
        self.seg_opts = seg_opts
        self.seg_opts['options'].setdefault('cellpose_options', {}).setdefault('min_size', 5000) # Set min_size to 5000 if it wasn't set
        self.seg_hpc_opts = hpc_opts if seg_hpc_opts is None else seg_hpc_opts
        self.seg_jobs = None

        # polygon parameters
        self.polygon_opts = {} if polygon_opts is None else polygon_opts
        self.polygon_opts.setdefault('save_file_extension', 'geojson')
        self.polygon_opts.setdefault('alpha_inv_coeff', 4/3)
        self.polygon_opts.setdefault('separate_z_planes', True)
        self.polygon_hpc_opts = hpc_opts if polygon_hpc_opts is None else polygon_hpc_opts
        self.polygon_jobs = None
        
        # intermediate file paths
        self.set_intermediate_file_paths()

        # metadata dict of initial parameters
        self.meta_path = output_dir.joinpath('seg_meta.json')
        self.update_metadata()


    @abstractmethod
    def get_load_func(self):
        return

    @abstractmethod
    def get_load_args(self):
        return

    def set_intermediate_file_paths(self):
        """Define locations within the output directory to save intermediate and output files.
        If you add a step to this pipeline that generates a file, specify its 
        path here.
        """
        output_dir = self.output_dir
        self.regions_path = output_dir.joinpath('regions.json')
        self.seg_run_spec_path = output_dir.joinpath('seg_run_spec.pkl')
        self.polygon_run_spec_path = output_dir.joinpath('polygon_run_spec.pkl')
        self.tile_save_path = output_dir.joinpath('seg_tiles/')
        self.cid_path = output_dir.joinpath('segmentation.npy')
        self.spot_table_path = output_dir.joinpath('seg_spot_table.npz')
        self.cbg_path = output_dir.joinpath('cell_by_gene.h5ad')
        self.polygon_subsets_path = output_dir.joinpath('cell_polygons/')
        self.polygon_final_path = self.output_dir.joinpath(f'cell_polygons.{self.polygon_opts["save_file_extension"]}')

    def update_metadata(self):
        """Update the metadata dictionary of segmentation input parameters.

        It is recommended to call this method whenever a function changes one of the
        SegmentationPipeline attributes set upon intialization.
        """
        self.meta = {
                'dt_file': self.detected_transcripts_file,
                'image_path': self.image_path,
                'output_dir': self.output_dir,
                'dt_cache': self.detected_transcripts_cache,
                'subrgn': self.subrgn,
                'seg_method': self.seg_method,
                'seg_opts': self.seg_opts,
                'polygon_opts': self.polygon_opts,
                'seg_hpc_opts': self.seg_hpc_opts,
                'polygon_hpc_opts': self.polygon_hpc_opts,  # hpc_opts not needed here
            }

    def save_metadata(self, overwrite=False):
        """Save SegmentationPipeline metadata to a json file in the output directory.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the current metadata file.
        """
        if not overwrite and self.meta_path.exists():
            raise FileExistsError('Metadata already saved and overwriting is not enabled.')

        else:
            # Make entries compatible with json
            metadata_cl = self.meta.copy()
            for k, v in self.meta.items():
                if isinstance(v, PurePath):
                    metadata_cl[k] = v.as_posix()
                elif inspect.isclass(v):
                    metadata_cl[k] = v.__module__ + '.' + v.__name__
                elif not isinstance(v, str|tuple|dict):
                    metadata_cl[k] = str(v)

            with open(self.meta_path, 'w') as f:
                json.dump(metadata_cl, f)

    def load_metadata(self):
        """Load the current SegmentationPipeline metadata json into a dictionary.

        Returns
        -------
        dict
            SegmentationPipeline attributes and their values as stored in the metadata file.
        """
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
 
        return meta

    def save_regions(self, regions: list, overwrite=False):
        """Save the segmentation tile subregion coordinates into a json file.

        Parameters
        ----------
        regions : list
            The list of subregion coordinates for every tile.
        overwrite : bool, optional
            Whether to overwrite the regions json file if it exists in the output directory.
        """
        regions_df = pd.DataFrame(regions, columns=['xlim', 'ylim'])
        if not overwrite and self.regions_path.exists():
            raise FileExistsError('Regions are already saved and overwriting is not enabled.')

        else:
            regions_df.to_json(self.regions_path)

    def load_regions(self):
        """Load the subregion coordinates from the output directory.

        Returns
        -------
        list
            The subregion coordinates for every tile.
        """
        assert self.regions_path.exists()
        regions = pd.read_json(self.regions_path).values
        regions = [tuple(r) for r in regions]
        return regions

    def save_run_spec(self, run_spec: dict, run_spec_path: Path, overwrite: bool=False):
        """Save run specifications for a job on the HPC as a pkl file.

        Parameters
        ----------
        run_spec : dict
            The run specifications.
        run_spec_path : Path
            The path to the run spec.
        overwrite : bool, optional
            Whether to enable overwriting of the run spec. Default False.
        """
        if not overwrite and run_spec_path.exists():
            raise FileExistsError('Run spec already saved and overwriting is not enabled.')

        else:
            with open(run_spec_path, 'wb') as f:
                pickle.dump(run_spec, f)

    def load_run_spec(self, run_spec_path: Path|str):
        """Load run specifications for a job on the HPC.

        Parameters
        ----------
        run_spec_path : Path, str, optional
            File path to the run spec.

        Returns
        -------
        dict
            The subregion coordinates for every tile.
        """
        with open(run_spec_path, 'rb') as f:
            run_spec = pickle.load(f)

        return run_spec

    def save_cell_ids(self, cell_ids, overwrite=False):
        """Save array of cell_ids to an npy file."""
        if not overwrite and self.cid_path.exists():
            raise FileExistsError('Cell ids already saved and overwriting is not enabled.')

        else:
            np.save(self.cid_path, cell_ids)

    def load_cell_ids(self):
        """Load array of cell_ids from an npy file."""
        assert self.cid_path.exists()
        cell_ids = np.load(self.cid_path)
        return cell_ids

    def save_seg_spot_table(self, overwrite=False):
        if not overwrite and self.spot_table_path.exists():
            raise FileExistsError('Segmented spot table already saved and overwriting is not enabled.')

        else:
            self.seg_spot_table.save_npz(self.spot_table_path)

    def load_seg_spot_table(self, allow_pickle=True):
        assert self.spot_table_path.exists()
        seg_spot_table = SegmentedSpotTable.load_npz(self.spot_table_path, allow_pickle=allow_pickle)
        return seg_spot_table

    def save_cbg(self, cell_by_gene, overwrite=False):
        """Save the cell by gene anndata object."""
        if not overwrite and self.cbg_path.exists():
            raise FileExistsError('Cell by gene already saved and overwriting is not enabled.')

        else:
            # geojson objects must be converted to strings before saving
            for k, v in cell_by_gene.uns.items():
                if isinstance(v, geojson.feature.FeatureCollection) or isinstance(v, geojson.geometry.GeometryCollection):
                    cell_by_gene.uns[k] = geojson.dumps(v)
            cell_by_gene.write(self.cbg_path)

    def load_cbg(self):
        """Load the cell by gene anndata object."""
        assert self.cbg_path.exists()
        cell_by_gene = ad.read_h5ad(self.cbg_path)
        return cell_by_gene

    def load_raw_spot_table(self):
        """Load the raw spot table, crop it by subregion. and set as an
        attribute.
        """
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        
        if isinstance(self.subrgn, str):
            subrgn = table.get_image(channel=self.subrgn).bounds()

        else:
            subrgn = self.subrgn 

        subtable = table.get_subregion(xlim=subrgn[0], ylim=subrgn[1])
        self.raw_spot_table = subtable

    def run(self, x_format: str, prefix: str='', suffix: str='', overwrite: bool=False, clean_up: str|bool|None='all_ints', tile_size: int=200, min_transcripts: int=0, rerun: bool=True):
        """Run all steps to perform tiled segmentation.

        Parameters
        ----------
        x_format: str
            Desired format for the cell by gene anndata X. Options: 'dense' or
            'sparse'.
        prefix: str, optional
            The string to prepend to all production cell ids.
        suffix: str, optional
            The string to append to all production cell ids.
        overwrite: bool, optional
            Whether to allow overwriting of output files. Default False.
        clean_up: str, bool, None, optional
            Whether or not to clean up intermediate files after segmentation
            Accepts: 'all_ints', 'seg_ints', 'polygon_ints', 'none', True, False, None
            Default: cleans up all intermediate files.
        tile_size: int, optional
            The maximum size of tiles to segment. Default 200. Increasing this
            parameter may also require increasing time and/or memory limits in
            seg_hpc_opts.
        min_transcripts : int, optional
            Minimum number of transcripts in a tile to be considered for segmentation. Default 0.
        rerun: bool, optional
            If enabled, SegmentationPipeline will attempt to automatically rerun jobs that failed
            If job failed due to memory constaints, memory limit in will be doubled
            If job failed due to time constaints, time limit in will be doubled

        Returns
        -------
        sis.spot_table.SegmentedSpotTable
            The segmented spot table.
        anndata.AnnData
            The cell by gene table.
        """

        # update and save run metadata in case user updated parameters
        self.update_metadata()
        self.save_metadata(overwrite)

        # load the raw spot table corresponding to the segmentation region
        self.load_raw_spot_table()

        # run all steps in sequence
        tiles, regions = self.tile_seg_region(overwrite=overwrite, max_tile_size=tile_size, min_transcripts=min_transcripts)
        seg_run_spec = self.get_seg_run_spec(regions=regions, overwrite=overwrite, result_files=False if clean_up else True)
        self.seg_jobs = self.submit_jobs('segmentation', seg_run_spec, overwrite)
        if rerun:
            self.seg_jobs = self.rerun_failed_jobs('segmentation_rerun', self.seg_jobs, seg_run_spec)
        cell_ids, merge_results, seg_skipped = self.merge_segmented_tiles(run_spec=seg_run_spec, tiles=tiles, overwrite=overwrite)
        polygon_run_spec = self.get_polygon_run_spec(overwrite)
        self.polygon_jobs = self.submit_jobs('cell_polygons', polygon_run_spec, overwrite)
        if rerun:
            self.polygon_jobs = self.rerun_failed_jobs('cell_polygons_rerun', self.polygon_jobs, polygon_run_spec)
        cell_polygons, cell_polygons_skipped = self.merge_cell_polygons(run_spec=polygon_run_spec, overwrite=overwrite)
        cell_by_gene = self.create_cell_by_gene(x_format=x_format, prefix=prefix, suffix=suffix, overwrite=overwrite)

        self.save_seg_spot_table(overwrite=overwrite)

        if clean_up:
            clean_up = 'all_ints' if clean_up == True else clean_up # If the user decides to input true we'll just set that to all ints
            self.clean_up(clean_up)

        return self.seg_spot_table, cell_by_gene

    def resume(self):
        raise NotImplementedError('Resuming from previous segmentation not implemented.')

    def track_job_progress(self, jobs: SlurmJobArray):
        """Track progress of submitted hpc jobs until all jobs have ended.

        Parameters
        ----------
        jobs : sis.hpc.SlurmJobArray
            Submitted slurm jobs to track.
        """
        print(f'Job IDs: {jobs[0].job_id}-{jobs[-1].job_id.split("_")[-1]}')
        with tqdm(total=len(jobs)) as pbar:
            while not jobs.is_done():
                time.sleep(60)
                n_done = int(np.sum([1 for job in jobs.jobs if job.is_done()]))
                pbar.update(n_done - pbar.n)
        if not np.any([job.state().state == "COMPLETED" for job in jobs.jobs]):
            raise RuntimeError(f'All jobs failed. Please check error logs in {self.output_dir.joinpath("hpc-jobs")}')

    def tile_seg_region(self, overwrite: bool=False, max_tile_size: int=200, overlap: int=30, min_transcripts=0):
        """Split the attached SpotTable into rectangular subregions (tiles).
        Also saves the subregion coordinates into a json file.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the regions json file (if it exists).
        max_tile_size : int
            Maximum width and height of the tiles in microns. Default 200.
        overlap : int
            Amount of overlap between tiles in microns. Default 30.
        min_transcripts : int
            Minimum number of transcripts in a tile to be considered for segmentation. Default 0.
        
        Returns
        -------
        list of sis.spot_table.SpotTable
            The grid of overlapping tiles.
        list
            Subregion coordinates for each tile.
        """
        print('Tiling segmentation region...')
        subtable = self.raw_spot_table

        tiles = subtable.grid_tiles(max_tile_size=max_tile_size, overlap=overlap, min_transcripts=min_transcripts)
        regions = [tile.parent_region for tile in tiles]

        # save regions
        self.save_regions(regions, overwrite)

        return tiles, regions

    def get_seg_run_spec(self, regions: list|None=None, overwrite: bool=False, result_files: bool=True):
        """Create a run specification for segmenting tiles on the HPC.

        Parameters
        ----------
        regions : list, optional
            The list of subregion coorindates for every tile. If not provided,
            will attempt to load subregion coordinates from disk.
        overwrite : bool, optional
            Whether to overwrite the run_spec file if it exists in the output directory.
        result_files : bool, optional
            Whether to save the spot table tiles as individual pickle files. Default True.
            Recommended to set to False if wanting to save disk space.

        Returns
        -------
        dict
            The segmentation run specifications.
        """
        if regions is None:
            regions = self.load_regions()
        self.tile_save_path.mkdir(exist_ok=True)

        print(f"Generating segmentation spec for {len(regions)} tiles...")
        run_spec = {}
        for i, region in enumerate(regions):
            run_spec[i] = (
                run_segmentation,
                (),
                dict(
                    load_func=self.get_load_func(),
                    load_args=self.get_load_args(),
                    subregion=region,
                    method_class=self.seg_method,
                    method_args=self.seg_opts,
                    result_file=os.path.join(f'{self.tile_save_path.as_posix()}/', f'segmentation_result_{i}.pkl') if result_files else None,
                    cell_id_file=os.path.join(f'{self.tile_save_path.as_posix()}/', f'segmentation_result_{i}.npy'),
                )
            )
        
        # save run_spec
        self.save_run_spec(run_spec, self.seg_run_spec_path, overwrite)

        return run_spec

    def _check_overwrite_files(self, run_spec, overwrite_file_keys, overwrite):
        """Helper function to check whether to overwrite files in a directory
        
        Parameters
        ----------
        run_spec: dict  
            The run specifications used to submit the jobs which will output files that we may want to overwrite
        overwrite_file_keys: list[str]
            List of key-names to access in run_spec's kwargs dict to check for overwriting.
        overwrite: bool
            Whether to overwrite result files
        """
        if overwrite: # If we are allowed to overwrite, just return
            return
        
        files_to_check = [v[2][k] for v in run_spec.values() for k in overwrite_file_keys]
        for file in files_to_check:
            if file is not None and os.path.exists(file):
                raise FileExistsError(f'Saved {file} file detected in directory and overwriting is disabled.')
    

    def submit_jobs(self, job_type: str, run_spec: dict|None=None, overwrite: bool=False):
        """Submit array jobs to the HPC.

        Parameters
        ----------
        job_type : str
            The type of jobs to submit. Set to 'segmentation' to run tiled segmentation or 'cell_polygons' to calculate cell polygons.
        run_spec : dict, optional
            The specifications to run the jobs on the HPC. If not provided, will attempt to load from the standard location on disk.
        overwrite : bool, optional
            Whether to overwrite result files. Default False.

        Returns
        -------
        sis.hpc.SlurmJobArray
            Object representing submitted HPC jobs.
        """
        # Check job type and set variables
        if 'segmentation' in job_type:
            if run_spec is None:
                run_spec = self.load_run_spec(self.seg_run_spec_path)
            hpc_opts = self.seg_hpc_opts
            self._check_overwrite_files(run_spec, ['result_file', 'cell_id_file'], overwrite)
            # Set defaults
            hpc_opts.setdefault('mem', '20G')
            hpc_opts.setdefault('time', '00:30:00')
            hpc_opts.setdefault('gpus_per_node', 1)
            status_str = 'Segmenting tiles...'
        elif 'cell_polygons' in job_type:
            if run_spec is None:
                run_spec = self.load_run_spec(self.polygon_run_spec_path)
            hpc_opts = self.polygon_hpc_opts
            self._check_overwrite_files(run_spec, ['result_file'], overwrite)
            # Set defaults
            hpc_opts.setdefault('mem', '10G')
            hpc_opts.setdefault('time', '00:30:00')
            hpc_opts.setdefault('gpus_per_node', None)
            status_str = 'Calculating cell polygons...'
        else:
            raise ValueError('Invalid job type.') 
        
        job_path = self.output_dir.joinpath('hpc-jobs')
        job_path.mkdir(exist_ok=True)
        hpc_opts.update({'job_path': f'{job_path.as_posix()}/'})

        hpc_config = {
            'run_spec': run_spec,
            'conda_env': hpc_opts['conda_env'],
            'hpc_host': hpc_opts['hpc_host'],
            'job_path': job_path,
            'partition': 'celltypes',
            'job_name': job_type,
            'nodes': 1,
            'ntasks': 1,
            'array': f'0-{len(run_spec)-1}',
            'mincpus': 1,
            'mail_user': None,
        }

        hpc_config.update(**hpc_opts)
        jobs = run_slurm_func(**hpc_config)
        print(status_str)
        self.track_job_progress(jobs)
        return jobs
    
    def merge_segmented_tiles(self, run_spec: dict|None=None, tiles: list[SegmentedSpotTable]|None=None, overwrite: bool=False):
        """Merge segmented tiles to generate and save the array of cell_ids.
        A new SegmentedSpotTable is created from the raw SpotTable and 
        updated with cell ids in place.

        Parameters
        ----------
        run_spec : dict, optional
            Specifications to run tiled segmentation on the HPC.
            If not provided, will attempt to load from the standard location on disk.
        tiles: list of sis.spot_table.SegmentedSpotTable, optional
            The individual tiles that were segmented.
            If not provided, will be generated from spot_table and run_spec.
        overwrite : bool, optional
            Whether to overwrite the cell ids file.
        
        Returns
        -------
        numpy.ndarray
            The array of cell_ids corresponding to each spot.
        list
            Information about merge conflicts collected during tile merging.
        list
            Indices of tiles skipped during segmentation.
        """
        if run_spec is None:
            run_spec = self.load_run_spec(self.seg_run_spec_path)

        print('Merging tiles...')
        # Merging updates the spot table cell_ids in place

        # tuples cannot be saved in anndata object, so convert to list
        # this is an issue if gauss or median kernels are specified
        seg_opts = self.seg_opts
        seg_opts = convert_value_nested_dict(seg_opts, tuple, list)

        truncated_meta = {
                'seg_method': str(self.seg_method),
                'seg_opts': seg_opts,
                'polygon_opts': self.polygon_opts
                }

        self.seg_spot_table = SegmentedSpotTable(
                spot_table=self.raw_spot_table, 
                cell_ids=np.empty(len(self.raw_spot_table), dtype=int),
                seg_metadata=truncated_meta,
                )
        self.seg_spot_table.cell_ids[:] = -1

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
                tile = self.raw_spot_table.get_subregion(xlim = tile_rgn[0], ylim = tile_rgn[1])
            
            tile_cids = np.load(cell_id_file)
            tile = SegmentedSpotTable(tile, tile_cids)
            result = self.seg_spot_table.merge_cells(tile, padding=5)
            merge_results.append(result)

        cell_ids = self.seg_spot_table.cell_ids

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save cell_ids
        self.save_cell_ids(cell_ids, overwrite)

        return cell_ids, merge_results, skipped

    def get_polygon_run_spec(self, overwrite: bool=False):
        """Generates a cell polygon run spec for running cell polygon jobs on the HPC.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the run spec if it exists. Default False.

        Returns
        -------
        dict
            The polygon run specifications.
        """
        self.polygon_subsets_path.mkdir(exist_ok=True)

        # Find all the cell ids
        unique_cells = np.unique(self.seg_spot_table.cell_ids)
        unique_cells = np.delete(unique_cells, np.where((unique_cells == 0) | (unique_cells == -1)))
        num_cells = len(unique_cells)

        # list of tuples assigning cells to jobs
        if 'num_jobs' in self.polygon_hpc_opts:
            num_jobs = self.polygon_hpc_opts.pop('num_jobs')
            row_list = [(i * num_cells // num_jobs, (i + 1) * num_cells // num_jobs) for i in range(num_jobs)]
        elif 'num_cells_per_job' in self.polygon_hpc_opts:
            import math
            num_cells_per_job = self.polygon_hpc_opts.pop('num_cells_per_job')
            row_list = [(i * num_cells_per_job, min((i + 1) * num_cells_per_job, num_cells)) for i in range(math.ceil(num_cells / num_cells_per_job))]
        else:
            num_jobs = 100 # default to 100 jobs
            row_list = [(i * num_cells // num_jobs, (i + 1) * num_cells // num_jobs) for i in range(num_jobs)]
        
        print(f"Generating cell polygon spec for {len(row_list)} jobs...")
        run_spec = {}
        for i, (start_idx, end_idx) in enumerate(row_list):
            # Save an input file with the cell IDs to calculate for each job
            if not overwrite and (self.polygon_subsets_path / f'cell_id_subset_{i}.npy').exists():
                raise FileExistsError(f'cell_id_subset_{i}.npy already exists and overwriting is not enabled.')
            np.save(self.polygon_subsets_path / f'cell_id_subset_{i}.npy', unique_cells[start_idx:end_idx])
            
            # run_spec[i] = (function, args, kwargs)
            run_spec[i] = (
                sis.spot_table.run_cell_polygon_calculation,
                (),
                dict(
                    load_func=self.get_load_func(),
                    load_args=self.get_load_args(),
                    subregion=self.subrgn,
                    cell_id_file=self.cid_path.as_posix(),
                    cell_subset_file=self.polygon_subsets_path.joinpath(f'cell_id_subset_{i}.npy').as_posix(),
                    result_file=self.polygon_subsets_path.joinpath(f'cell_polygons_subset_{i}.{self.polygon_opts["save_file_extension"]}').as_posix(),
                    alpha_inv_coeff=self.polygon_opts['alpha_inv_coeff'],
                    separate_z_planes=self.polygon_opts['separate_z_planes'],
                )
            )

        self.save_run_spec(run_spec, self.polygon_run_spec_path, overwrite)

        return run_spec

    def merge_cell_polygons(self, run_spec: dict|None=None, overwrite: bool=False):
        """Add cell polygons calculated across subsets of cells to the attached spot table in place.

        Parameters
        ----------
        run_spec : dict, optional
            The run specification for calculating polygons on the HPC. 
            If not provided, will attempt to load from the standard location on disk.
        overwrite : bool, optional
            Whether to overwrite the polygon run spec if it exists in the output directory. Default False.
        
        Returns
        -------
        dict
            The cell polygons after merging.
        list
            Indices of cell polygons that were skipped.
        """
        if run_spec is None:
            run_spec = self.load_run_spec(self.polygon_run_spec_path)

        print('Merging cell polygons...')
        skipped = []
        for i, area_spec in enumerate(tqdm(run_spec.values())):
            result_file = area_spec[2]['result_file']
            if not os.path.exists(result_file):
                print(f"Skipping tile {i} : no result file generated")
                skipped.append(i)
                continue

            self.seg_spot_table.load_cell_polygons(result_file, reset_cache=False, disable_tqdm=True) # The reset_cache=False is important to allow reading in the various cell subsets without overwriting

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save polygons
        if not overwrite and self.polygon_final_path.exists():
            raise FileExistsError('cell polygons already saved and overwriting is not enabled.')
        self.seg_spot_table.save_cell_polygons(self.output_dir / f'cell_polygons.{self.polygon_opts["save_file_extension"]}')

        return self.seg_spot_table.cell_polygons, skipped
    
    
    def create_cell_by_gene(self, x_format: str, prefix: str='', suffix: str='', overwrite: bool=False):
        """Create and save a cell by gene AnnData object from the attached
        spot table.
        
        Parameters
        ----------
        x_format: str
            Desired format for the cell by gene anndata X. Options: 'dense' or
            'sparse'.
        prefix: str, optional
            The string to prepend to all production cell ids.
        suffix: str, optional
            The string to append to all production cell ids.
        overwrite: bool, optional
            Whether to allow overwriting of output files. Default False.

        Returns
        -------
        anndata.AnnData
            The cell by gene table.
        """
        self.seg_spot_table.generate_production_cell_ids(prefix=prefix, suffix=suffix)
        cell_by_gene = self.seg_spot_table.cell_by_gene_anndata(x_format=x_format)
        self.save_cbg(cell_by_gene, overwrite)

        return cell_by_gene
        
    def clean_up(self, mode="all_ints"):
        """Clean up intermediate files after segmentation and polygon generation is complete.

        Parameters
        ----------
        mode : str, optional 
            Can be 'all_ints', 'seg_ints', 'polygon_ints', or 'none' depending on desired clean up. Defaults to 'all_ints'.
        """
        if mode not in ['all_ints', 'seg_ints', 'polygon_ints', 'none']:
            raise ValueError('Invalid clean up mode')
        if mode == "all_ints" or mode == "seg_ints":
            for file_path in self.tile_save_path.glob('*'):
                file_path.unlink()
            self.tile_save_path.rmdir()
        if mode == "all_ints" or mode == "polygon_ints":
            for file_path in self.polygon_subsets_path.glob('*'):
                file_path.unlink()
            self.polygon_subsets_path.rmdir()

    @classmethod
    def from_spatial_dataset(cls, sp_dataset, output_dir, subrgn, seg_method, seg_opts, polygon_opts=None, seg_hpc_opts=None, polygon_hpc_opts=None, hpc_opts=None):
        """Alternate constructor to load from a SpatialDataset"""
        image_path = sp_dataset.images_path
        csv_file = sp_dataset.detected_transcripts_file
        cache_file = sp_dataset.detected_transcripts_cache

        return cls(csv_file, image_path, output_dir, cache_file, subrgn, seg_method, seg_opts, hpc_opts, polygon_opts, seg_hpc_opts, polygon_hpc_opts, hpc_opts)

    @classmethod
    def from_json(cls, json_file):
        """Load a run from a json file"""
        with open(json_file, 'r') as f:
            config = json.load(f)
        seg_method_name = config['seg_method'].rpartition('.')[-1]
        if seg_method_name == 'CellposeSegmentationMethod':
            config['seg_method'] = sis.segmentation.CellposeSegmentationMethod
        else:
            raise NotImplementedError(f'Segmentation method {seg_method_name} not implemented.')
        if isinstance(config['subrgn'], list):
            config['subrgn'] = tuple([tuple(l) for l in config['subrgn']])

        return cls(**config)

    def rerun_failed_jobs(self, job_type: str, jobs: SlurmJobArray, run_spec: dict, mem: str|None=None, time: str|None=None, max_attempts: int=5):
        """
        This function takes jobs and the run_spec that submitted them and resubmits any failed jobs.
        It continues to resubmit until all jobs are completed properly or the maximum number of attempts is reached
        
        Parameters
        ----------
        job_type : str
            The type of jobs to rerun. Set to 'segmentation' to run tiled segmentation or 'cell_polygons' to calculate cell polygons.
        jobs: sis.hpc.SlurmJobArray
            A SlurmJobArray instance containing jobs to check for failures
        run_spec: dict
            The run_spec that was used to submit the jobs previously
        mem: str, optional
            The amount of memory that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's memory
        time: str, optional
            The length of time that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's time
        max_attempts: int, optional
            The maximum number of times to attempt to rerun the failed jobs. Default 5

        Returns
        -------
        The inputted SlurmJobArray with the completed rerun jobs inserted
        """
        indices_to_rerun, failure_types = self.find_failed_jobs(jobs)
        loops = 0
        while indices_to_rerun != None and loops < max_attempts:
            print(f'Resubmitting failed jobs...')
            print(f'Failed jobs: {",".join(str(x) for x in indices_to_rerun)}')
            print(f'Reasons: {",".join([k for k, v in failure_types.items() if v])}')
            new_jobs = self.resubmit_failed_jobs(job_type + f"_{loops}", indices_to_rerun, failure_types, run_spec, mem=mem, time=time)
            self.update_jobs(jobs, indices_to_rerun, new_jobs)
            indices_to_rerun, failure_types = self.find_failed_jobs(jobs)
            loops += 1
        return jobs
    
    def find_failed_jobs(self, jobs: SlurmJobArray):
        """
        This function takes a sis.hpc.SlurmJobArray and identifies which jobs—-if any--failed and how they failed
        
        Parameters
        ----------
        jobs: sis.hpc.SlurmJobArray
            A SlurmJobArray instance containing jobs to check for failures
        
        Returns
        -------
        If no jobs failed: None, None
        If some jobs failed:
            to_rerun: list[int]
                A list of the indices of the failed jobs in the inputted SlurmJobArray
            failure_types: dict
                A dictionary with keys as types of failues and bools representing if that failure occured in the inputted SlurmJobArray
        """
        job_state_dict = jobs.state()
        failure_types= {"OUT_OF_MEMORY": False, "TIMEOUT": False, "CANCELLED": False}
        failed_jobs_indices = []
        for job_index, job_state in enumerate(job_state_dict.values()):
            if job_state.state == "COMPLETED":
                continue
            elif job_state.state in failure_types.keys():
                failed_jobs_indices.append(job_index)
                failure_types[job_state.state] = True
            else:
                raise ValueError(f"Could not automatically rerun jobs. Job state must be one of OUT_OF_MEMORY, TIMEOUT, or CANCELLED. Job state was {job_state.state}")
        
        if len(failed_jobs_indices) == 0:
            return None, None# No jobs to rerun
        else:
            return failed_jobs_indices, failure_types
        
    def resubmit_failed_jobs(self, job_type: str, indices_to_rerun: list[int], failure_types: dict, run_spec: dict, mem: str|None=None, time: str|None=None):
        """
        This function resubmits failed jobs.
        It requires a list of indices to rerun, a dictionary of reasons the jobs failed, and the run_spec originally used to submit the jobs
        
        Parameters
        ----------
        job_type : str
            The type of jobs to resubmit. Set to 'segmentation' to run tiled segmentation or 'cell_polygons' to calculate cell polygons.
        indices_to_rerun: list[int]
            A list of the indices of the failed jobs
        failure_types: dict
            A dictionary with keys as types of failues and bools representing if that failure occured failed jobs
        run_spec: dict
            The run_spec that was used to submit the jobs previously
        mem: str, optional
            The amount of memory that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's memory
        time: str, optional
            The length of time that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's time

        Returns
        -------
        A sis.hpc.SlurmJobArray containing all the resubmitted jobs
        """
        if 'segmentation' in job_type:
            hpc_opts = self.seg_hpc_opts
        elif 'cell_polygons' in job_type:
            hpc_opts = self.polygon_hpc_opts
        else:
            raise ValueError('Invalid job type.')
        
        new_run_spec = {}
        for job_array_index, failed_tile_index in enumerate(indices_to_rerun):
            new_run_spec[job_array_index] = run_spec[failed_tile_index]
        
        if failure_types["OUT_OF_MEMORY"]:
            if mem is not None and memory_to_bytes(mem) > memory_to_bytes(hpc_opts["mem"]):
                new_mem = mem # If the user set memory is larger than the previously used memory set the memory to the user setting
            else: # Double the previously used memory if user did not set memory or user set memory is <= previously used time
                new_mem = double_mem(hpc_opts["mem"])
            
            if memory_to_bytes(new_mem) > memory_to_bytes('500GB'): # Cap the memory at 500GB
                print('Requested memory exceeds limit. Setting memory to 500GB.')
                new_mem = '500GB'
                
            hpc_opts["mem"] = new_mem
            print('New memory allocation:', hpc_opts["mem"])
        
        if failure_types["TIMEOUT"]:
            if time is not None and slurm_time_to_seconds(time) > slurm_time_to_seconds(hpc_opts["time"]):
                new_time = time # If the user set time is larger than the previously used time set the time to the user setting
            else: # Double the previously used time if user did not set time or user set time is <= previously used time
                new_time = seconds_to_time(slurm_time_to_seconds(hpc_opts["time"]) * 2)
            
            if slurm_time_to_seconds(new_time) > slurm_time_to_seconds('60:00:00'): # Cap the time at 60 hours
                print('Requested time exceeds limit. Setting time limit to 60 hours.')
                new_time = '60:00:00'
            
            hpc_opts["time"] = new_time
            print('New time limit:', hpc_opts["time"])
            
        return self.submit_jobs(job_type, new_run_spec, overwrite=True)

    def update_jobs(self, jobs, indices_to_replace, new_jobs):
        """
        This function takes a sis.hpc.SlurmJobArray, a list of indices to replace, and a secondary sis.hpc.SlurmJobArray
        It then replaces the jobs in the original array with the new ones, creating a franken-SlurmJobArray
        This goes against the general expected behavior of sis.hpc.SlurmJobArray but doesn't break any functionality used in keeping track of job status.
        It may cause inconsistencies in SlurmJobArray.args, SlurmJobArray.sbatch_output, SlurmJobArray.job_file, SlurmJobArray.host, or SlurmJobArray.job_id
        These inconsistencies will be limited to the SlurmJobArray class, if the user looks at indivual SlurmJobs in SlurmJobArray.jobs, they will all have correct information
        
        Parameters
        ----------
        jobs: sis.hpc.SlurmJobArray
            A SlurmJobArray instance containing jobs replace with new ones
        indices_to_rerun: list[int]
            A list of the indices of to replace
        new_jobs: sis.hpc.SlurmJobArray
            A SlurmJobArray containing the jobs which will replace those in 'jobs'
        
        Returns
        -------
        The updated SlurmJobArray
        """
        for new_jobs_idx, (jobs_idx, job) in enumerate(zip(indices_to_replace, new_jobs.jobs)):
            jobs.jobs[jobs_idx] = SlurmJob(job.args, job.sbatch_output, job.job_file, job.host, array_id=new_jobs_idx, job_array=jobs)
        return jobs


class MerscopeSegmentationPipeline(SegmentationPipeline):
    def __init__(
            self,
            dt_file: Path|str,
            image_path: Path|str,
            output_dir: Path|str,
            dt_cache: Path|str|None,
            subrgn: str|tuple,
            seg_method: SegmentationMethod,
            seg_opts: dict,
            polygon_opts: dict|None=None,
            seg_hpc_opts: dict|None=None,
            polygon_hpc_opts: dict|None=None,
            hpc_opts: dict|None=None
            ):
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, polygon_opts, seg_hpc_opts=seg_hpc_opts, polygon_hpc_opts=polygon_hpc_opts, hpc_opts=hpc_opts)

    def get_load_func(self):
        """Get the function to load a spot table."""
        return SpotTable.load_merscope

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
        load_args = {
                'image_path': self.image_path,
                'csv_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
        }
        for k, v in load_args.items():
            if isinstance(v, Path):
                load_args[k] = v.as_posix()

        load_args['max_rows'] = None

        return load_args


class StereoSeqSegmentationPipeline(SegmentationPipeline):
    def __init__(
            self,
            dt_file: Path|str,
            image_path: Path|str,
            output_dir: Path|str,
            dt_cache: Path|str|None,
            subrgn: str|tuple,
            seg_method: SegmentationMethod,
            seg_opts: dict,
            polygon_opts: dict|None=None,
            seg_hpc_opts: dict|None=None,
            polygon_hpc_opts: dict|None=None,
            hpc_opts: dict|None=None
            ):
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, polygon_opts, seg_hpc_opts=seg_hpc_opts, polygon_hpc_opts=polygon_hpc_opts, hpc_opts=hpc_opts)

    def get_load_func(self):
        """Get the function to load a spot table."""
        return SpotTable.load_stereoseq

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
        load_args = {
                'image_file': self.image_path,
                'gem_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
        }

        for k, v in load_args.items():
            if isinstance(v, Path):
                load_args[k] = v.as_posix()

        load_args.update({'skiprows': 7, 'image_channel': 'nuclear'})

        return load_args
