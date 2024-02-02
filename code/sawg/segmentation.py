from __future__ import annotations
import os, tempfile, pickle, traceback
import scipy.ndimage, scipy.interpolate
import numpy as np
from tqdm.autonotebook import tqdm
from .hpc import run_slurm_func
from .spot_table import SpotTable
from .image import Image, ImageBase, ImageTransform

import inspect
import json
from pathlib import Path, PurePath
import time
from abc import abstractmethod
import pandas as pd
import anndata as ad
import sawg


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
    """Base class for running segmentation on a whole section (or subregion).
    When the class is initialized, it creates the segmentation output directory
    and sets paths for intermediate files.

    Each step has defined input and output files. Steps can be run individually
    by calling their respective methods or in a defined sequence by calling the
    run() method.

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
        full region bounded by an associated image channel. To segment a 
        smaller region, set to a tuple corresponding to a bounding box.

    seg_method: SegmentationMethod
        The segmentation method to use. Must be found in sawg.segmentation.

    seg_opts: dict
        Options to pass to seg_method.

    hpc_opts: dict
        Options to use for segmenting tiles on the hpc.
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
            polygon_opts: dict,
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

        # intermediate file paths
        self.set_intermediate_file_paths()

        if (seg_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either seg_hpc_opts or hpc_opts must be provided.")
        if (polygon_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either polygon_hpc_opts or hpc_opts must be provided.")

        # segmentation parameters
        self.subrgn = subrgn
        self.seg_method = seg_method
        self.seg_opts = seg_opts
        self.seg_hpc_opts = hpc_opts if seg_hpc_opts is None else seg_hpc_opts

        # polygon parameters
        self.polygon_opts = polygon_opts
        self.polygon_hpc_opts = hpc_opts if polygon_hpc_opts is None else polygon_hpc_opts

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
        output_dir = self.output_dir
        self.regions_path = output_dir.joinpath('regions.json')
        self.seg_run_spec_path = output_dir.joinpath('seg_run_spec.pkl')
        self.polygon_run_spec_path = output_dir.joinpath('polygon_run_spec.pkl')
        self.tile_save_path = output_dir.joinpath('seg_tiles/')
        self.cid_path = output_dir.joinpath('segmentation.npy')
        self.cbg_path = output_dir.joinpath('cell_by_gene.h5ad')
        self.polygon_save_path = output_dir.joinpath('cell_polygons/')

    def update_metadata(self):
        self.meta = {
                'dt_file': self.detected_transcripts_file,
                'image_path': self.image_path,
                'output_dir': self.output_dir,
                'dt_cache': self.detected_transcripts_cache,
                'subrgn': self.subrgn,
                'seg_method': self.seg_method,
                'seg_opts': self.seg_opts,
                'seg_hpc_opts': self.seg_hpc_opts,
                'polygon_hpc_opts': self.polygon_hpc_opts,
            }

    def save_metadata(self, overwrite=False):
        if not overwrite and self.meta_path.exists():
            raise FileExistsError('Metadata already saved and overwriting is not enabled.')

        else:
            metadata_cl = self.meta.copy()
            for k, v in self.meta.items():
                if isinstance(v, PurePath):
                    metadata_cl[k] = v.as_posix()
                elif inspect.isclass(v):
                    metadata_cl[k] = v.__module__ + '.' + v.__name__
                elif not isinstance(v, str|tuple):
                    metadata_cl[k] = str(v)

            with open(self.meta_path, 'w') as f:
                json.dump(metadata_cl, f)

    def load_metadata(self):
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
 
        return meta

    def save_regions(self, regions, overwrite=False):
        regions_df = pd.DataFrame(regions, columns=['xlim', 'ylim'])
        if not overwrite and self.regions_path.exists():
            raise FileExistsError('Regions are already saved and overwriting is not enabled.')

        else:
            regions_df.to_json(self.regions_path)

    def load_regions(self):
        assert self.regions_path.exists()
        regions = pd.read_json(self.regions_path).values
        regions = [tuple(r) for r in regions]
        return regions

    def save_run_spec(self, run_spec, run_spec_path, overwrite=False):
        if not overwrite and run_spec_path.exists():
            raise FileExistsError('Run spec already saved and overwriting is not enabled.')

        else:
            with open(run_spec_path, 'wb') as f:
                pickle.dump(run_spec, f)

    def load_run_spec(self, run_spec_path):
        with open(run_spec_path, 'rb') as f:
            run_spec = pickle.load(f)

        return run_spec

    def save_cell_ids(self, cell_ids, overwrite=False):
        if not overwrite and self.cid_path.exists():
            raise FileExistsError('Cell ids already saved and overwriting is not enabled.')

        else:
            np.save(self.cid_path, cell_ids)

    def load_cell_ids(self):
        assert self.cid_path.exists()
        cell_ids = np.load(self.cid_path)
        return cell_ids

    def save_cbg(self, cell_by_gene, overwrite=False):
        if not overwrite and self.cbg_path.exists():
            raise FileExistsError('Cell by gene already saved and overwriting is not enabled.')

        else:
            cell_by_gene.write(self.cbg_path)

    def load_cbg(self):
        assert self.cbg_path.exists()
        cell_by_gene = ad.read_h5ad(self.cbg_path)
        return cell_by_gene

    def load_spot_table(self):
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        
        if isinstance(self.subrgn, str):
            subrgn = table.get_image(channel=self.subrgn).bounds()

        else:
            subrgn = self.subrgn 

        subtable = table.get_subregion(xlim=subrgn[0], ylim=subrgn[1])

        return subtable

    def run(self, use_prod_cids: bool=True, prefix: str='', suffix: str='', overwrite: bool=False, clean_up=True):
        """Run all steps to perform tiled segmentation.

        Parameters
        ----------
        use_prod_cids: bool, optional
            If True, generate production cell ids and use them to index cells
            in the cell by gene table. Default True.

        prefix: str, optional
            The string to prepend to all production cell ids.

        suffix: str, optional
            The string to append to all production cell ids.

        overwrite: bool, optional
            Whether to allow overwriting of output files. Default False.

        Returns
        -------
        SpotTable
            The segmented spot table.

        AnnData
            The cell by gene table.
        """

        # update and save run metadata in case user updated parameters
        self.update_metadata()
        self.save_metadata(overwrite)

        # load the spot table corresponding to the segmentation region
        self.spot_table = self.load_spot_table()

        # run all steps in sequence
        tiles, regions = self.tile_seg_region(overwrite)
        seg_run_spec = self.get_seg_run_spec(regions=regions, overwrite=overwrite, result_files=False if clean_up else True)
        jobs = self.submit_jobs('segmentation', seg_run_spec, overwrite)
        cell_ids, merge_results, seg_skipped = self.merge_segmented_tiles(run_spec=seg_run_spec, overwrite=overwrite)
        polygon_run_spec = self.get_polygon_run_spec(overwrite)
        jobs = self.submit_jobs('cell_polygons', polygon_run_spec, overwrite)
        cell_polygons, cell_polygons_skipped = self.merge_cell_polygons(run_spec=polygon_run_spec, overwrite=overwrite)
        cell_by_gene = self.create_cell_by_gene(use_prod_cids=use_prod_cids, prefix=prefix, suffix=suffix, overwrite=overwrite)

        if clean_up:
            self.clean_up()

        return self.spot_table, cell_by_gene

    def resume(self):
        raise NotImplementedError('Resuming from previous segmentation not implemented.')

    def load_results(self):
        """Load the results of a finished segmentation."""
        self.spot_table.cell_ids = self.load_cell_ids()
        cell_by_gene = self.load_cbg()
        return self.spot_table, cell_by_gene

    def track_job_progress(self, jobs):
        """Track progress of hpc jobs. until all jobs have ended
        """
        print(f'Job IDs: {jobs[0].job_id}-{jobs[-1].job_id.split("_")[-1]}')
        with tqdm(total=len(jobs)) as pbar:
            while not jobs.is_done():
                time.sleep(60)
                n_done = int(np.sum([1 for job in jobs.jobs if job.is_done()]))
                pbar.update(n_done - pbar.n)
        if not np.any([job.state().state == "COMPLETED" for job in jobs.jobs]):
            raise RuntimeError(f'All jobs failed. Please check error logs in {self.output_dir.joinpath("hpc-jobs")}')

    def tile_seg_region(self, overwrite=False, max_tile_size: int=200, overlap: int=30):
        print('Tiling segmentation region...')
        subtable = self.spot_table

        tiles = subtable.grid_tiles(max_tile_size=max_tile_size, overlap=overlap)
        regions = [tile.parent_region for tile in tiles]

        # save regions
        self.save_regions(regions, overwrite)

        return tiles, regions

    def get_seg_run_spec(self, regions=None, overwrite=False, result_files=True):
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

    def _check_overwrite_dir(self, overwrite_dir, overwrite_file, overwrite):
        """Helpeer function to check whether to overwrite files in a directory"""
        if overwrite and any(overwrite_dir.glob(overwrite_file)):
            print(f'Deleting saved {overwrite_file} files from previous run...')
            for tile_path in overwrite_dir.glob(overwrite_file):
                if tile_path.is_file():
                    tile_path.unlink()
        elif not overwrite and any(overwrite_dir.glob(overwrite_file)):
            raise FileExistsError(f'Saved {overwrite_file} files detected in directory and overwriting is disabled.')

    def submit_jobs(self, job_type, run_spec=None, overwrite=False):
        # Check job type and set variables
        if job_type == 'segmentation':
            run_spec_path = self.seg_run_spec_path 
            hpc_opts = self.seg_hpc_opts
            self._check_overwrite_dir(self.tile_save_path, 'segmentation_result*', overwrite)
            default_mem = "20G"
            gpus_per_node = 1
            check_file = 'cell_id_file'
            status_str = 'Segmenting tiles...'
        elif job_type == 'cell_polygons':
            run_spec_path = self.polygon_run_spec_path
            hpc_opts = self.polygon_hpc_opts
            self._check_overwrite_dir(self.polygon_save_path, 'cell_polygons_subset_*', overwrite)
            default_mem = "10G"
            gpus_per_node = None
            check_file = 'result_file'
            status_str = 'Calculating cell polygons...'
        else:
            raise ValueError('Invalid job type.') 
        
        if run_spec is None:
            run_spec = self.load_run_spec(run_spec_path)
        
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
            'gpus_per_node': gpus_per_node,
            'mem': default_mem,
            'time': '0:30:00',
            'mail_user': None,
        }

        hpc_config.update(**hpc_opts)
        jobs = run_slurm_func(**hpc_config)
        print(status_str)
        self.track_job_progress(jobs)

        return jobs

    def merge_segmented_tiles(self, run_spec=None, overwrite=False):
        if run_spec is None:
            run_spec = self.load_run_spec(self.seg_run_spec_path)

        print('Merging tiles...')
        # Merging updates the spot table cell_ids in place
        cell_ids, merge_results, skipped = merge_segmentation_results(self.spot_table, run_spec, None)

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save cell_ids
        self.save_cell_ids(cell_ids, overwrite)

        return cell_ids, merge_results, skipped

    def get_polygon_run_spec(self, overwrite=False):
        """ Generates a cell polygon run spec for running cell polygon jobs on the hpc """
        self.polygon_save_path.mkdir(exist_ok=True)

        if overwrite and any(self.polygon_save_path.glob('cell_id_subset_*.npy')):
            print('Deleting save id subsets from previous run...')
            for tile_path in self.tile_save_path.glob('cell_id_subset_*.npy'):
                if tile_path.is_file():
                    tile_path.unlink()
        elif not overwrite and any(self.polygon_save_path.glob('cell_id_subset_*.npy')):
            raise RuntimeError('Saved id subsets detected in directory and overwriting is disabled.')

        # Find all the cell ids
        unique_cells = np.unique(self.spot_table.cell_ids)
        unique_cells = np.delete(unique_cells, np.where((unique_cells == 0) | (unique_cells == -1)))
        num_cells = len(unique_cells)

        # list of tuples assigning cells to jobs
        num_jobs = self.polygon_opts['num_jobs']
        row_list = [(i * num_cells // num_jobs, (i + 1) * num_cells // num_jobs) for i in range(num_jobs)]
        
        save_file_extension = self.polygon_opts['save_file_extension'] if self.polygon_opts['save_file_extension'] else 'geojson'
        alpha_inv_coeff = self.polygon_opts['alpha_inv_coeff'] if self.polygon_opts['alpha_inv_coeff'] else 4/3

        print(f"Generating cell polygon spec for {len(row_list)} jobs...")
        run_spec = {}
        for i, (start_idx, end_idx) in enumerate(row_list):
            # Save an input file with the cell IDs to calculate for each job
            np.save(self.polygon_save_path / f'cell_id_subset_{i}.npy', unique_cells[start_idx:end_idx])
            
            # run_spec[i] = (function, args, kwargs)
            run_spec[i] = (
                sawg.spot_table.run_cell_polygon_calculation,
                (),
                dict(
                    load_func=self.get_load_func(),
                    load_args=self.get_load_args(),
                    cell_id_file=self.cid_path,
                    cell_subset_file=self.polygon_save_path / f'cell_id_subset_{i}.npy',
                    result_file=self.polygon_save_path / f'cell_polygons_subset_{i}.{save_file_extension}',
                    alpha_inv_coeff=alpha_inv_coeff,
                )
            )

        self.save_run_spec(run_spec, self.polygon_run_spec_path, overwrite)

        return run_spec

    def merge_cell_polygons(self, run_spec=None, overwrite=False):
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

            self.spot_table.load_cell_polygons(result_file, reset_cache=False, disable_tqdm=True) # The reset_cache=False is important to allow reading in the various cell subsets without overwriting

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save polygons
        save_file_extension = self.polygon_opts['save_file_extension'] if self.polygon_opts['save_file_extension'] else 'geojson'
        if not overwrite and self.output_dir / (f'cell_polygons.{save_file_extension}').exists():
            raise FileExistsError('cell polygons already saved and overwriting is not enabled.')
        self.spot_table.save_cell_polygons(self.output_dir / f'cell_polygons.{save_file_extension}')
        
        return self.spot_table.cell_polygons, skipped
    
    
    def create_cell_by_gene(self, remove_bg=True, use_prod_cids=True, prefix='', suffix='', overwrite=False):
        """Create and save a cell by gene file in Anndata format using 
        the attached spot table.
        """
        if not use_prod_cids and (prefix != '' or suffix != ''):
            print('Warning: Prefix and/or suffix have been set, but production cell ids are not being used.')

        if use_prod_cids:
            self.spot_table.generate_production_cell_ids(prefix=prefix, suffix=suffix)

        subtable = self.spot_table.filter_cells(real_cells=remove_bg)

        cell_by_gene = subtable.cell_by_gene_anndata(use_both_ids=use_prod_cids)
        
        # Calculate cell volumes to add to cell by gene
        print('Calculating cell volumes...')
        cell_feature_df = subtable.get_cell_features().sort_values(by='cell_id')
        cell_by_gene.obs['volume'] = cell_feature_df['volume']
        
        self.save_cbg(cell_by_gene, overwrite)

        return cell_by_gene
        
    def clean_up(self, segmentation=True, polygons=True):
        """ Clean up intermediate files after segmentation and polygon generation is complete

        Args:
            polygons (bool, optional): Clean up intermediate polygon files. Defaults to True.
            segmentation (bool, optional): Clean up intermediate segmentation files. Defaults to True.
        """
        if segmentation:
            for file_path in self.tile_save_path.glob('*'):
                file_path.unlink()
            self.tile_save_path.rmdir()
        if polygons:
            for file_path in self.polygon_save_path.glob('*'):
                file_path.unlink()
            self.polygon_save_path.rmdir()

    @classmethod
    def from_spatial_dataset(cls, sp_dataset, output_dir, subrgn, seg_method, seg_opts, hpc_opts):
        """Alternate constructor to load from a SpatialDataset"""
        image_path = sp_dataset.images_path
        csv_file = sp_dataset.detected_transcripts_file
        cache_file = sp_dataset.detected_transcripts_cache

        return cls(csv_file, image_path, output_dir, cache_file, subrgn, seg_method, seg_opts, hpc_opts)

    @classmethod
    def from_json(cls, json_file):
        """Load a run from a json file"""
        with open(json_file, 'r') as f:
            config = json.load(f)
        seg_method_name = config['seg_method'].rpartition('.')[-1]
        if seg_method_name == 'CellposeSegmentationMethod':
            config['seg_method'] = sawg.segmentation.CellposeSegmentationMethod
        else:
            raise NotImplementedError(f'Segmentation method {seg_method_name} not implemented.')
        if isinstance(config['subrgn'], list):
            config['subrgn'] = tuple([tuple(l) for l in config['subrgn']])

        return cls(**config)


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
            polygon_opts: dict,
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
            polygon_opts: dict,
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
                'image_file': self.images_path,
                'gem_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
        }

        for k, v in load_args.items():
            if isinstance(v, Path):
                load_args[k] = v.as_posix()

        load_args.update({'skiprows': 7, 'image_channel': 'nuclear'})

        return load_args
