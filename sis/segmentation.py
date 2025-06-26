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

def run_segmentation(load_func, load_args:dict, method_class, method_args:dict, subregion:dict|None=None, result_file:str|None=None, cell_id_file:str|None=None):
    """Load a spot table, run segmentation (possibly on a subregion), and save the SegmentationResult.

    Parameters
    ----------
    load_func : function
        The method of SpotTable used to load a dataset (e.g. SpotTable.load_merscope).
    load_args : dict
        Parameters passed to load_func.
    method_class : SegmentationMethod
        The SegmentationMethod used for segmentation.
    method_args : dict
        The arguments to pass to method_class.
    subregion : dict or None, optional
        The subregion of the SpotTable to segment.
    result_file : str or None, optional
        Where to save the SegmentationResult object.
    cell_id_file : str or None, optional
        Where to save the list of cell ids output by segmentation.
    """
    spot_table = load_func(**load_args)
    print(f"loaded spot table {len(spot_table)}")
    if subregion is not None:
        spot_table = spot_table.get_subregion(*subregion)
    print(f"subregion {subregion} {len(spot_table)}")
    seg = method_class(method_args)
    result = seg.run(spot_table)
    print(f"cell_ids {len(result.cell_ids)}")

    if result_file is not None:
        result.save(result_file)
        print(f"saved segmentation result to {result_file}")
    if cell_id_file is not None:
        np.save(cell_id_file, result.cell_ids)
        print(f"saved segmentated cell IDs to {cell_id_file}")


class SegmentationResult:
    """Base class defining a segmentation of SpotTable data--method, options, results
        
    Attributes
    ----------
    method : SegmentationMethod
        The segmentation method used.
    input_spot_table : sis.spot_table.SpotTable
        The input spot table upon which segmentation was run
    """
    
    def __init__(self, method: SegmentationMethod, input_spot_table: SpotTable):
        """
        Parameters
        ----------
        method : SegmentationMethod
            The segmentation method used.
        input_spot_table : sis.spot_table.SpotTable
            The input spot table upon which segmentation was run
        """
        self.method = method
        self.input_spot_table = input_spot_table

    @property
    def cell_ids(self):
        """Array of segmented cell IDs for each spot in the table
        """
        raise NotImplementedError()        

    def spot_table(self, min_spots: int|None=None):
        """Return a SegmentedSpotTable with cell_ids determined by the segmentation.

        Parameters
        ----------
        min_spots : int or None, optional
            The minimum number of spots required for a cell to be considered valid. Cells with fewer spots will be assigned a cell ID of 0.
        """
        cell_ids = self.cell_ids

        if min_spots is not None:
            # Get rid of cells with fewer than min_spots
            cell_ids = cell_ids.copy()
            cids, counts = np.unique(cell_ids, return_counts=True)
            for cid, count in zip(cids, counts):
                if count < min_spots:
                    mask = cell_ids == cid
                    cell_ids[mask] = 0

        return SegmentedSpotTable(self.input_spot_table, cell_ids)

    def save(self, filename):
        """Save the SegmentationResult to a pickle file.
        """
        pickle.dump(self, open(filename, 'wb'))


class SegmentationMethod:
    """Base class defining segmentation methods.

    Subclasses should initialize with a dictionary of options, then calling
    run(spot_table) will execute the segmentation method and return a SegmentationResult.
    
    Attributes
    ----------
    options : dict
        The options for the segmentation method.
    """
    def __init__(self, options:dict):
        """
        Parameters
        ----------
        options : dict
            The options for the segmentation method.
        """
        self.options = options
        
    def run(self, spot_table:SpotTable):
        """Dependent on the method, run segmentation on the spot table and return a SegmentationResult
        """
        raise NotImplementedError()
        
    def _get_spot_table(self, spot_table:SpotTable):
        """Return the SpotTable instance to run segmentation on. 
        
        If spot_table is a string, load from npz file.
        If a sub-region is specified in options, return the sub-table.
        
        Parameters
        ----------
        spot_table : SpotTable or str
            The spot table to run segmentation on.
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
    
    Example
    -------
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
        
    Attributes
    ----------
    options : dict
        The options for the segmentation method. (example detailed above)
    """
    
    def __init__(self, options):
        """
        Parameters
        ----------
        options : dict
            The options for the segmentation method. (example detailed above)
        """
        super().__init__(options)
        
    def run(self, spot_table):
        """Run cellpose segmentation on a spot table and return a CellposeSegmentationResult
        
        Specifications as to the nature of the segmentation are specified in self.options (of particular note is 'cellpose_model' and 'images').
        
        Parameters
        ----------
        spot_table : SpotTable
            The spot table to run segmentation on.
            
        Raises
        ------
        ValueError
            If 'detect_z_planes' is specified in options and 'frame' is also specified in the options['images']
        """
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
        cp_opts.update(self.options['cellpose_options']) # Update defaults with options provided by user
        if self.options['cell_dia'] is not None: # cell diameter must be converted to pixels from um
            manual_diam = self.options['cell_dia'] / self.options['px_size']
            cp_opts.update({'diameter': manual_diam})
        
        # Limit the z-planes to be segmented by checking transcript z distribution
        if 'detect_z_planes' in self.options and self.options['detect_z_planes']:
            z_planes = spot_table.detect_z_planes(float_cut=self.options['detect_z_planes'])
            if 'nuclei' in self.options['images']:
                if 'frame' in self.options['images']['nuclei']:
                    raise ValueError('Only one of "detect_z_planes" or "frame" may be specified')
                self.options['images']['nuclei']['frames'] = z_planes
            if 'cyto' in self.options['images']:
                if 'frame' in self.options['images']['cyto']:
                    raise ValueError('Only one of "detect_z_planes" or "frame" may be specified')
                self.options['images']['cyto']['frames'] = z_planes

        # collect images from spot table / generate total mRNA image
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
        cp_opts['do_3D'] = list(images.values())[0].shape[0] > 1
        if cp_opts['do_3D']: # Only need to use anisotropy if we are segmenting in 3d
            cp_opts.setdefault('anisotropy', self.options['z_plane_thickness'] / self.options['px_size'])
        if len(images) == 2:
            assert images['cyto'].shape == images['nuclei'].shape

            # If the number of z-planes is too small we want to double the z-planes while still maintaining order
            if self.options.get('duplistack', False):
                cyto_data = CellposeSegmentationMethod.duplistack_image(images['cyto'])
                nuclei_data = CellposeSegmentationMethod.duplistack_image(images['nuclei'])
            else:
                cyto_data = images['cyto'].get_data()
                nuclei_data = images['nuclei'].get_data()
                
            image_data = np.empty((cyto_data.shape[:3]) + (3,), dtype=cyto_data.dtype)
            image_data[..., 0] = cyto_data
            image_data[..., 1] = nuclei_data
            image_data[..., 2] = 0
            channels = [1, 2]  # cyto=1 (red), nuclei=2 (green)
        else:
            # If the number of z-planes is too small we want to double the z-planes while still maintaining order
            if self.options.get('duplistack', False):
                image_data = CellposeSegmentationMethod.duplistack_image(list(images.values())[0]) # Since there is only one image, we just take the first value
            else:
                image_data = list(images.values())[0].get_data()
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

        # If we made a duplistacked image lets return it back to the original size
        if self.options.get('duplistack', False):
            cellpose_output['masks'] = cellpose_output['masks'][::2]
            cellpose_output['flows'][0] = cellpose_output['flows'][0][::2]
            cellpose_output['flows'][1] = cellpose_output['flows'][1][:, ::2]
            cellpose_output['flows'][2] = cellpose_output['flows'][2][::2]
            cellpose_output['flows'][3] = cellpose_output['flows'][3][:, ::2]

        # Increase size of masks if spcified by user
        dilate = self.options.get('dilate', 0)
        if dilate != 0:
            masks = dilate_labels(masks, radius=dilate/self.options['px_size'])
            cellpose_output.update({'masks': masks})

        # return result object
        result = CellposeSegmentationResult(
            method=self, 
            input_spot_table=spot_table,
            cellpose_output=cellpose_output,
            image_transform=list(images.values())[0].transform,
            detect_z_planes=self.options['detect_z_planes'] if 'detect_z_planes' in self.options and self.options['detect_z_planes'] else None,
        )
            
        return result

    def _read_image_spec(self, img_spec, spot_table, px_size, images):
        """Return an Image to be used in segmentation based on img_spec:
        
        Parameters
        ----------
        img_spec : str or dict or Image
            - An Image instance is returned as-is
            - "total_mrna" returns an image generated from spot density
            - Any other string returns an image channel attached to the spot table
            - {'channel': channel, 'frame': int} can be used to select a single frame
            - {'channel': 'total_mrna', 'n_planes': int, 'frame': int, 'gauss_kernel': (1, 3, 3), 'median_kernel': (2, 10, 10)} can be ued to configure total mrna image generation
        spot_table : sis.spot_table.SpotTable
            The spot table which is either used to create the image or already contains the image
        px_size : float
            The size of pixels in the image
        images : dict
            A dictionary containing images to be segmentated
            
        Returns
        -------
        sis.image.Image
            An image to be used for segmentation
            
        Raises
        ------
        TypeError
            If img_spec is not a string, dict, or ImageBase instance.
        """
        # optionally, cyto image may be generated from spot table total mrna        
        if img_spec is None or isinstance(img_spec, ImageBase):
            return img_spec
        if isinstance(img_spec, str):
            img_spec = {'channel': img_spec}
        if not isinstance(img_spec, dict):
            raise TypeError(f"Bad image spec: {img_spec}")

        if img_spec['channel'] == 'total_mrna':
            # We have to create the total mRNA image from the spot table
            opts = img_spec.copy()
            opts.pop('channel')
            opts.update(self._suggest_image_spec(spot_table, px_size, images))
            return self.get_total_mrna_image(spot_table, **opts)
        else:
            # Assume the image is already present in the spot table
            # Support both multiple frames and individual frames
            return spot_table.get_iamge(channel=img_spec['channel'],
                                        frames=img_spec.get('frames', img_spec.get('frame', None)))


    def _suggest_image_spec(self, spot_table, px_size, images):
        """Given a pixel size, return {'image_shape': shape, 'image_transform': tr} covering the entire area of spot_table.
        
        If any images are already present, use those as templates instead.
        
        Parameters
        ----------
        spot_table : sis.spot_table.SpotTable
            The spot table which is used to define the transformation matrix
        px_size : float
            The size of pixels in the image
        images : dict
            A dictionary containing images to be segmentated
                
        Returns
        -------
        dict
            A dictionary containing the shape and transformation matrix of the image
        """
        if len(images) > 0: # If there are images already present, use the first one as a template
            img = list(images.values())[0]
            return {'image_shape': img.shape[:3], 'image_transform': img.transform}

        # Find the pixel shape from the transcript information
        bounds = np.array(spot_table.bounds())
        scale = 1 / px_size
        shape = np.ceil((bounds[:, 1] - bounds[:, 0]) * scale).astype(int)
        
        # create a transformation matrix that maps the spot coordinates to pixel coordinates
        tr_matrix = np.zeros((2, 3))
        tr_matrix[0, 0] = tr_matrix[1, 1] = scale
        tr_matrix[:, 2] = -scale * bounds[:, 0]
        image_tr = ImageTransform(tr_matrix)
        
        return {'image_shape': (1, shape[0], shape[1]), 'image_transform': image_tr}

    def get_total_mrna_image(self, spot_table, image_shape:tuple, image_transform:ImageTransform, n_planes:int, frames:tuple|int|None=None, gauss_kernel=(1, 3, 3), median_kernel=(2, 10, 10)):
        """Create a total mRNA image (histogram of spot density) from the spot table.
       
        Can be used to approximate cytosol staining for segmentation.
        Smoothing can optionally be applied.

        Parameters
        ----------
        spot_table : sis.spot_table.SpotTable
            The spot table used to create the image.
        image_shape : tuple
            The shape of the image.
        image_transform : ImageTransform
            The transform that relates image and spot coordinates (?).
        n_planes : int
            The number of z planes in the image.
        frames : tuple or int or None, optional
            A tuple of the first (inclusive) and last (exclusive) indices of the frames (specific z planes) used to create the image, e.g. frames=(2,5) would create an image from z planes 2, 3, and 4.
            else, if an int is provided, that specific z plane is used.
        gauss_kernel : tuple, optional
            Kernel used for gaussian smoothing of the image. Default (1, 3, 3).
        median_kernel : tuple, optional
            Kernel used for median smoothing of the image. Default (2, 10, 20).
            
        Returns
        -------
        Image
            The total mRNA image.
        """

        image_shape_full = (n_planes, *image_shape[1:3])
        density_img = np.zeros(image_shape_full, dtype='float32')

        # map spots to pixel coordinates
        spot_px = self.map_spots_to_img_px(spot_table, image_transform=image_transform, image_shape=image_shape_full)
        for i in range(n_planes): 
            # Calculate a 2D histogram of spot positions for each z plane
            # Have to do every z-plane even with frame or frames specified since kernels can take cross plane information
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

        # Create sis.Image class from the density image and get the frames specified
        return Image(density_img[..., np.newaxis], transform=image_transform, channels=['Total mRNA'], name=None).get_frames(frames)

    def map_spots_to_img_px(self, spot_table: SpotTable, image: Image|None=None, image_transform: ImageTransform|None=None, image_shape: tuple|None=None, detect_z_planes: float|None=None):
        """Map spot table (x, y, z) positions to image pixels (frame, row, col). 
        
        Optionally, you can provide the *image_transform* and *image_shape* in place of an *image*.
        
        Parameters
        ----------
        spot_table : SpotTable
            The spot table to map to image pixels.
        image : Image or None, optional
            The image to use for mapping.
        image_transform : ImageTransform or None, optional
            The transform that relates image and spot coordinates.
            Can be used in conjunction with image_shape as a replacement for image.
        image_shape : tuple or None, optional
            The shape of the image.
            Can be used in conjunction with image_transform as a replacement for image.
        detect_z_planes : float or None, optional
            If provided, limit the z-planes to those that contain at least *detect_z_planes* fraction of spots.
            
        Returns
        -------
        np.ndarray
            An array of shape (n_spots, 3) containing the (frame, row, col) positions of each spot.
        """
        # Only one of image and (image_transform, image_shape) should be provided
        if image is not None:
            assert image_transform is None and image_shape is None
            image_shape = image.shape
            image_transform = image.transform

        if detect_z_planes:
            # Limit the z-planes to assign to pixels (useful for xenium data where a lot of z-planes are empty)
            z_planes = spot_table.detect_z_planes(float_cut=detect_z_planes)
            z_mask = np.isin(spot_table.z, [z for z in range(*z_planes)])
            spot_table = spot_table[z_mask]

        spot_xy = spot_table.pos[:, :2]
        spot_px_rc = np.floor(image_transform.map_to_pixels(spot_xy)).astype(int)

        # for this dataset, z values are already integer index instead of um
        if spot_table.z is None:
            spot_px_z = np.zeros(len(spot_table), dtype=int)
        else:
            spot_px_z = spot_table.z.astype(int)
            if detect_z_planes: # If we are limiting the z-planes, we need to shift the z values to start at 0
                spot_px_z -= z_planes[0]

        # some spots may be a little past the edge of the image; 
        # just clip these as they'll be discarded when tiles are merged anyway
        spot_px_rc[:, 0] = np.clip(spot_px_rc[:, 0], 0, image_shape[1]-1)
        spot_px_rc[:, 1] = np.clip(spot_px_rc[:, 1], 0, image_shape[2]-1)
        
        return np.hstack([spot_px_z[:, np.newaxis], spot_px_rc])

    @staticmethod
    def duplistack_image(image: Image):
        """This method takes an image and creates a numpy array of its data duplicated in place
        
        e.g. an image with frames 1,2,3 would become an array with frames 1,1,2,2,3,3

        Parameters
        ----------
        image : Image
            The image to duplistack.
        """
        img_data = image.get_data()
        frames, rows, cols = img_data.shape[:3]
        duplistacked_img_data = np.zeros((frames * 2, rows, cols), dtype=img_data.dtype)
        for i in range(frames):
            # Duplicate each frame and place it next to itself
            duplistacked_img_data[2*i] = img_data[i]
            duplistacked_img_data[2*i+1] = img_data[i]
            
        return duplistacked_img_data
        

class CellposeSegmentationResult(SegmentationResult):
    """Class made for containing the result of running CellposeSegmentationMethod on a SpotTable
    
    Attributes
    ----------
    method : SegmentationMethod
        The segmentation method used.
    input_spot_table : sis.spot_table.SpotTable
        The input spot table upon which segmentation was run
    cellpose_output : dict
        The output of cellpose segmentation. Contains
            - 'masks'
            - 'flows'
            - 'styles'
            - 'diams' (if non-custom model).
    image_transform : sis.image.ImageTransform
        The transform that relates image and spot coordinates.
    _cell_ids : np.ndarray or None
        The cell IDs assigned to each spot in the table, or None if not yet computed.
    detect_z_planes : float or None
        If float limit the z-planes to those that contain at least *detect_z_planes* fraction of spots.
    """
    def __init__(self, method:SegmentationMethod, input_spot_table:SpotTable, cellpose_output:dict, image_transform:ImageTransform, detect_z_planes: float|None=None):
        """
        Parameters
        ----------
        method : sis.segmentation.SegmentationMethod
            The segmentation method used.
        input_spot_table : sis.spot_table.SpotTable
            The input spot table upon which segmentation was run
        cellpose_output : dict
            The output of cellpose segmentation. Contains 'masks', 'flows', 'styles', and 'diams' (if non-custom model).
        image_transform : sis.image.ImageTransform
            The transform that relates image and spot coordinates.
        detect_z_planes : float or None, optional
            If provided, limit the z-planes to those that contain at least *detect_z_planes* fraction of spots.
        """
        super().__init__(method, input_spot_table)
        self.cellpose_output = cellpose_output
        self.image_transform = image_transform
        self._cell_ids = None
        self.detect_z_planes = detect_z_planes
        
    @property
    def cell_ids(self):
        """Create an array of segmented cell IDs for each spot in the table from the mask image
        
        Returns
        -------
        np.ndarray
        """
        # use segmented masks to assign each spot to a cell
        if self._cell_ids is None:
            spot_table = self.input_spot_table
            mask_img = self.mask_image
            spot_px = self.method.map_spots_to_img_px(spot_table, mask_img, detect_z_planes=self.detect_z_planes)

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
        
        Returns
        -------
        sis.image.Image
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
        
    def spot_table(self, min_spots=None):
        """Return a SegmentedSpotTable with cell_ids determined by the segmentation.

        Parameters
        ----------
        min_spots : int or None, optional
            The minimum number of spots required for a cell to be considered valid.
            Cells with fewer spots will be assigned a cell ID of 0.
        """
        cell_ids = self.cell_ids

        # Remove cells with fewer than *min_spots* transcripts
        if min_spots is not None:
            cell_ids = cell_ids.copy()
            cids, counts = np.unique(cell_ids, return_counts=True)
            for cid, count in zip(cids, counts):
                if count < min_spots:
                    mask = cell_ids == cid
                    cell_ids[mask] = 0

        if self.detect_z_planes:
            # The cell IDs will not contain all z-planes, since some were ignored during segmentation
            # Thus we have to construct and use this same mask when setting cell IDs in the output spot table
            return_table = SegmentedSpotTable(self.input_spot_table, np.zeros(len(self.input_spot_table), dtype=int))
            z_plane_mask = self.input_spot_table.z_plane_mask(self.input_spot_table.detect_z_planes(float_cut=self.detect_z_planes))
            return_table.cell_ids[z_plane_mask] = cell_ids
            return return_table
        else:
            return SegmentedSpotTable(self.input_spot_table, cell_ids)
        

def dilate_labels(img, radius):
    """Dilate labeled regions of an image.

    Given an image with 0 in the background and objects labeled with different integer values (such
    as a cell segmentation mask), return a new image with objects expanded by *radius*.

    (Credit: https://stackoverflow.com/a/70261747)
    
    Parameters
    ----------
    img : np.ndarray
        The input image with labeled regions.
    radius : float
        The radius by which to dilate the labeled regions.
    
    Returns
    -------
    interpolated : np.ndarray
        The dilated image.
        
    Raises
    ------
    TypeError
        If the input image is not a 2D or 3D array.
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

    Notes
    -----
    Currently there is an assumption that the attributes set upon initialization
    are not changed by the user. If you change them, be warned that the 
    metadata dictionary and spot table subregion may become outdated if the 
    appropriate methods are not called afterward.
    
    Attributes
    ----------
    image_path : Path or str
        Path to image/directory containing images to be segmented.
    detected_transcripts_file : Path or str
        Path to transcipts file
    detected_transcripts_cache : Path or str or None
        Path to a npz detected transcripts file. Used for faster loading.
    output_dir : Path
        Path to the directory where all results will be saved.
    subrgn : str or tuple
        The subregion to segment. A string (e.g. 'DAPI') indicates segmentation
        of the full region bounded by the associated image channel. To segment a 
        smaller region, set to a tuple corresponding to a bounding box.
    seg_method : sis.segmentation.SegmentationMethod
        The segmentation method to use. Must be found in sis.segmentation.
    seg_opts : dict
        The argmuents to pass to the segmentation method.
    seg_hpc_opts : dict
        Options to use for segmenting tiles on an hpc
    seg_jobs : sis.hpc.SlurmJobArray or None
        Contains information on the jobs used to run tiled segmentation on the hpc.
    polygon_opts : dict
        Options to pass to sis.spot_table.run_cell_polygon_calculation()
    polygon_hpc_opts : dict
        Options to use for calculating polygons on an hpc
    polygon_jobs : sis.hpc.SlurmJobArray or None
        Contains information on the jobs used to run polygon calculations on the hpc.
    meta_path : Path
        Path to file containing initial parameters and settings.
    self.regions_path : Path
        Path to file containing tile boundary information
    seg_run_spec_path : Path
        Path to file containing segmentation run specifications
    polygon_run_spec_path : Path
        Path to file containing polygon run specifications
    tile_save_path : Path
        Path to directory where segmentation tile intermediate results are saved
    cid_path : Path
        Path to file containing cell IDs for each spot in the segmentation
    seg_spot_table_path : Path
        Path to final segmented spot table file (npz format)
    cbg_path : Path
        Path to final cell-by-gene table file (h5ad format)
    polygon_subsets_path : Path
        Path to directory where cell polygon intermediate results are saved
    polygon_final_path : Path
        Path to final cell polygon file (geojson or other format specified in polygon_opts)
    """
    def __init__(self, dt_file: Path|str, image_path: Path|str, output_dir: Path|str, dt_cache: Path|str|None, subrgn: str|tuple, seg_method: SegmentationMethod, seg_opts: dict, polygon_opts: dict|None=None, seg_hpc_opts: dict|None=None, polygon_hpc_opts: dict|None=None, hpc_opts: dict|None=None):
        """
        Parameters
        ----------
        dt_file : str or Path
            Path to the detected transcripts file.
        image_path : str or Path
            Path to the images.
        output_dir : str or Path
            Where to save output files.
        dt_cache : str or Path or None
            Path to the detected transcripts cache file. Used for faster loading.
        subrgn : str or tuple
            The subregion to segment. Set to a string, e.g. 'DAPI', to segment the 
            full region bounded by the associated image channel. To segment a 
            smaller region, set to a tuple corresponding to a bounding box.
        seg_method : SegmentationMethod
            The segmentation method to use. Must be found in sis.segmentation.
        seg_opts : dict
            Options to pass to seg_method.
        polygon_opts : dict or None, optional
            Options to pass to for cell polygon generation. Currently supports save_file_extension, alpha_inv_coeff, and separate z-planes.
            Default is None, which sets save_file_extension to 'geojson', alpha_inv_coeff to 4/3, and separate_z_planes to True   
        seg_hpc_opts : dict or None, optional
            Options to use for segmenting tiles on the hpc. Default is None
        polygon_hpc_opts : dict or None, optional
            Options to use for calculating cell polygons on the hpc. Default is None
        hpc_opts : dict or None, optional
            Options to use for both segmenting tiles and calculating cell polygons on the hpc (can be used in place of submitting both seg_hpc_opts and polygon_hpc_opts).
            Default is None
            
        Raises
        ------
        ValueError
            If neither seg_hpc_opts nor hpc_opts are provided, or if neither polygon_hpc_opts nor hpc_opts are provided.
        """

        # input/output paths
        self.image_path = image_path
        self.detected_transcripts_file = dt_file
        self.detected_transcripts_cache = dt_cache 

        # Make sure the path exists and is a Path object
        if isinstance(output_dir, str) or isinstance(output_dir, PurePath):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # The hpc options can either be general or split by segmentation and polygon generation
        if (seg_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either seg_hpc_opts or hpc_opts must be provided.")
        if (polygon_hpc_opts is None and hpc_opts is None):
            raise ValueError("One of either polygon_hpc_opts or hpc_opts must be provided.")

        # segmentation parameters
        self.subrgn = subrgn
        self.seg_method = seg_method
        self.seg_opts = seg_opts
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
        """Get the function to load a spot table."""
        return

    @abstractmethod
    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
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
        self.seg_spot_table_path = output_dir.joinpath('seg_spot_table.npz')
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
        """Save SegmentationPipeline metadata to the meta_path attribute (a json file in the output directory).

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the current metadata file.
            
        Raises
        ------
        FileExistsError
            If the metadata file already exists and overwrite is False.
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
        """Load the current SegmentationPipeline metadata json from the meta_path attribute into a dictionary.

        Returns
        -------
        dict
            SegmentationPipeline attributes and their values as stored in the metadata file.
        """
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        return meta

    def save_regions(self, regions: list, overwrite: bool=False):
        """Save the segmentation tile subregion coordinates into the regions_path attribute.

        Parameters
        ----------
        regions : list
            The list of subregion coordinates for every tile.
        overwrite : bool, optional
            Whether to overwrite the regions json file if it exists in the output directory.
            
        Raises
        ------
        FileExistsError
            If the regions file already exists and overwrite is False.
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
        regions : list
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
        
        Raises
        ------
        FileExistsError
            If the run spec file already exists and overwrite is False.
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
        run_spec_path : Path or str
            File path to the run spec.
        
        Returns
        -------
        dict
            The run specifications
        """
        with open(run_spec_path, 'rb') as f:
            run_spec = pickle.load(f)
        return run_spec

    def save_cell_ids(self, cell_ids: np.ndarray, overwrite: bool=False):
        """Save array of cell_ids to the cid_path attribute.
        
        Parameters
        ----------
        cell_ids : np.ndarray
            The array of cell IDs.
        overwrite : bool, optional
            Whether to enable overwriting of the cell ids file. Default False.
            
        Raises
        ------
        FileExistsError
            If the cell ids file already exists and overwrite is False.
        """
        if not overwrite and self.cid_path.exists():
            raise FileExistsError('Cell ids already saved and overwriting is not enabled.')
        else:
            np.save(self.cid_path, cell_ids)

    def load_cell_ids(self):
        """Load array of cell_ids from the cid_path attribute
        """
        assert self.cid_path.exists()
        cell_ids = np.load(self.cid_path)
        return cell_ids

    def save_seg_spot_table(self, overwrite: bool=False):
        """Save the segmented spot table to the seg_spot_table_path attribute.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to enable overwriting of the segmented spot table file. Default False.
            
        Raises
        ------
        FileExistsError
            If the segmented spot table file already exists and overwrite is False.
        """
        if not overwrite and self.seg_spot_table_path.exists():
            raise FileExistsError('Segmented spot table already saved and overwriting is not enabled.')
        else:
            self.seg_spot_table.save_npz(self.seg_spot_table_path)

    def load_seg_spot_table(self, allow_pickle: bool=True):
        """Load the segmented spot table from the seg_spot_table_path attribute.
        
        Parameters
        ----------
        allow_pickle : bool, optional
            Whether to allow loading pickled object arrays stored in npy files.
            Must be enabled to load dictionaries and cell polygons.
        """
        assert self.seg_spot_table_path.exists()
        seg_spot_table = SegmentedSpotTable.load_npz(self.seg_spot_table_path, allow_pickle=allow_pickle)
        return seg_spot_table

    def save_cbg(self, cell_by_gene: ad.AnnData, overwrite: bool=False):
        """Save the cell by gene anndata object to the cbg_path attribute.
        
        Parameters
        ----------
        cell_by_gene : anndata.AnnData
            The cell by gene anndata object.
        overwrite : bool, optional
            Whether to enable overwriting of the cell by gene file. Default False.
            
        Raises
        ------
        FileExistsError
            If the cell by gene file already exists and overwrite is False.
        """
        if not overwrite and self.cbg_path.exists():
            raise FileExistsError('Cell by gene already saved and overwriting is not enabled.')
        else:
            # geojson objects must be converted to strings before saving
            for k, v in cell_by_gene.uns.items():
                if isinstance(v, geojson.feature.FeatureCollection) or isinstance(v, geojson.geometry.GeometryCollection):
                    cell_by_gene.uns[k] = geojson.dumps(v)
                    
            # tuples cannot be saved in anndata object, so convert to str
            # this is an issue if gauss or median kernels are specified
            cell_by_gene.uns = convert_value_nested_dict(cell_by_gene.uns, tuple, str)
            
            cell_by_gene.write(self.cbg_path)

    def load_cbg(self):
        """Load the cell by gene anndata object from the cbg_path attribute
        """
        assert self.cbg_path.exists()
        cell_by_gene = ad.read_h5ad(self.cbg_path)
        return cell_by_gene

    def load_raw_spot_table(self):
        """Load the raw spot table [self.get_load_func(), self.get_load_args()], 
        crop it by subregion [self.subrgn],
        and set as an attribute. [self.raw_spot_table]
        """
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        
        # Subregion can be a channel name or tuple. If it's a channel name, get the bounds of the channel
        if isinstance(self.subrgn, str):
            subrgn = table.get_image(channel=self.subrgn).bounds()
        else:
            subrgn = self.subrgn

        subtable = table.get_subregion(xlim=subrgn[0], ylim=subrgn[1])
        self.raw_spot_table = subtable

    def run(self, x_format: str, x_dtype: str='uint16', additional_obs: dict|None=None, prefix: str='', suffix: str='', overwrite: bool=False, clean_up: str|bool|None='all_ints', tile_size: int=200, min_transcripts: int=0, rerun: bool=True):
        """Run all steps to perform tiled segmentation.

        Parameters
        ----------
        x_format: str
            Desired format for the cell by gene anndata X. Options: 'dense' or
            'sparse'.
        x_dtype : str, optional
            The data type of the matrix.
        additional_obs : dict or None, optional
            Additional columns to add to the anndata.obs DataFrame.
            Keys are column names and values are arrays of the same length as the number of cells.
        prefix: str, optional
            The string to prepend to all cell labels
        suffix: str, optional
            The string to append to all cell labels
        overwrite: bool, optional
            Whether to allow overwriting of output files. Default False.
        clean_up: str or bool or None, optional
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
        tuple[sis.spot_table.SegmentedSpotTable, anndata.AnnData]
            - The segmented spot table.
            - The cell by gene table.
        """

        # update and save run metadata in case user updated parameters
        self.update_metadata()
        self.save_metadata(overwrite)

        # load the raw spot table corresponding to the segmentation region
        self.load_raw_spot_table()

        # Segment the raw spot table
        tiles, regions = self.tile_seg_region(overwrite=overwrite, max_tile_size=tile_size, min_transcripts=min_transcripts)
        seg_run_spec = self.get_seg_run_spec(regions=regions, overwrite=overwrite, result_files=False if clean_up else True)
        self.seg_jobs = self.submit_jobs('segmentation', seg_run_spec, overwrite)
        if rerun:
            self.seg_jobs = self.rerun_failed_jobs('segmentation_rerun', self.seg_jobs, seg_run_spec)
        cell_ids, merge_results, seg_skipped = self.merge_segmented_tiles(run_spec=seg_run_spec, tiles=tiles, detect_z_planes=self.seg_opts.get('detect_z_planes', None), overwrite=overwrite)
        
        # Generate polygons for identified cells
        polygon_run_spec = self.get_polygon_run_spec(overwrite)
        self.polygon_jobs = self.submit_jobs('cell_polygons', polygon_run_spec, overwrite)
        if rerun:
            self.polygon_jobs = self.rerun_failed_jobs('cell_polygons_rerun', self.polygon_jobs, polygon_run_spec)
        cell_polygons, cell_polygons_skipped = self.merge_cell_polygons(run_spec=polygon_run_spec, overwrite=overwrite)
        
        # Create output cell-by-gene
        cell_by_gene = self.create_cell_by_gene(x_format=x_format, x_dtype=x_dtype, additional_obs=additional_obs, prefix=prefix, suffix=suffix, overwrite=overwrite)

        self.save_seg_spot_table(overwrite=overwrite)

        if clean_up:
            clean_up = 'all_ints' if clean_up == True else clean_up # If the user decides to input true we'll just set that to all ints
            self.clean_up(clean_up)

        return self.seg_spot_table, cell_by_gene

    def resume(self):
        raise NotImplementedError('Resuming from previous segmentation not implemented.')

    def track_job_progress(self, jobs: SlurmJobArray):
        """Track progress of submitted hpc jobs and display with tqdm until all jobs have ended.

        Parameters
        ----------
        jobs : sis.hpc.SlurmJobArray
            Submitted slurm jobs to track.
            
        Raises
        ------
        RuntimeError
            If none of the jobs completed properly, indicating a ubiquitous issue that must be resolved.
        """
        print(f'Job IDs: {jobs[0].job_id}-{jobs[-1].job_id.split("_")[-1]}')
        with tqdm(total=len(jobs)) as pbar:
            while not jobs.is_done():
                # Wait until all jobs are done to return, checking every 60 seconds
                time.sleep(60)
                n_done = int(np.sum([1 for job in jobs.jobs if job.is_done()]))
                pbar.update(n_done - pbar.n)
        # If none of the jobs completed properly we don't want to continue/rerun because it implies a ubiquitous issue which must be resolved
        if not np.any([job.state().state == "COMPLETED" for job in jobs.jobs]):
            raise RuntimeError(f'All jobs failed. Please check error logs in {self.output_dir.joinpath("hpc-jobs")}')

    def tile_seg_region(self, overwrite: bool=False, max_tile_size: int=200, overlap: int=30, min_transcripts=0):
        """Split the attached SpotTable into rectangular subregions (tiles).
        Also saves the subregion coordinates into a json file.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the regions json file (if it exists).
        max_tile_size : int, optional
            Maximum width and height of the tiles in microns. Default 200.
        overlap : int, optional
            Amount of overlap between tiles in microns. Default 30.
        min_transcripts : int, optional
            Minimum number of transcripts in a tile to be considered for segmentation. Default 0.
        
        Returns
        -------
        (tiles, regions) : tuple[list[SpotTable], list[tuple]]
            - The grid of overlapping tiles.
            - Subregion coordinates for each tile.
        """
        print('Tiling segmentation region...')
        subtable = self.raw_spot_table

        # Use grid tiles because we want tiles of equal size, not transcript count
        tiles = subtable.grid_tiles(max_tile_size=max_tile_size, overlap=overlap, min_transcripts=min_transcripts)
        regions = [tile.parent_region for tile in tiles]

        # save regions
        self.save_regions(regions, overwrite)

        return tiles, regions

    def get_seg_run_spec(self, regions: list|None=None, overwrite: bool=False, result_files: bool=True):
        """Create a run specification for segmenting tiles on the HPC.

        Parameters
        ----------
        regions : list or None, optional
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
            # run_spec[i] = (function, args, kwargs)
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

    def _check_overwrite_files(self, run_spec: dict, overwrite_file_keys: list[str], overwrite: bool):
        """Helper function to check whether to overwrite files in a directory
        
        Parameters
        ----------
        run_spec : dict  
            The run specifications used to submit the jobs which will output files that we may want to overwrite
        overwrite_file_keys : list[str]
            List of key-names to access in run_spec's kwargs dict to check for overwriting.
        overwrite : bool
            Whether to overwrite result files
            
        Raises
        ------
        FileExistsError
            If the a file that already exists is trying to be written to the directory and overwrite is False.
        """
        if overwrite: # If we are allowed to overwrite, just return
            return
        
        # Checking run_spec args (idx=2) for user-specified kwargs containing files that we may want to overwrite
        files_to_check = [v[2][k] for v in run_spec.values() for k in overwrite_file_keys]
        for file in files_to_check:
            if file is not None and os.path.exists(file):
                raise FileExistsError(f'Saved {file} file detected in directory and overwriting is disabled.')
    

    def submit_jobs(self, job_type: str, run_spec: dict|None=None, overwrite: bool=False):
        """Submit array jobs to a SLURM managed HPC.

        Parameters
        ----------
        job_type : str
            The type of jobs to submit. Set to 'segmentation' to run tiled segmentation or 'cell_polygons' to calculate cell polygons.
        run_spec : dict or None, optional
            The specifications to run the jobs on the HPC. If not provided, will attempt to load from the standard location on disk.
        overwrite : bool, optional
            Whether to overwrite result files. Default False.
        
        Returns
        -------
        sis.hpc.SlurmJobArray
            Object representing submitted HPC jobs.
            
        Raises
        ------
        ValueError
            If an invalid job type is specified. (i.e. not 'segmentation' or 'cell_polygons').
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

    def merge_segmented_tiles(self, run_spec: dict|None=None, tiles: list[SegmentedSpotTable]|None=None, detect_z_planes: float|None=None, overwrite: bool=False):
        """Merge segmented tiles to generate and save the array of cell_ids.
        A new SegmentedSpotTable is created from the raw SpotTable and 
        updated with cell ids in place.

        Parameters
        ----------
        run_spec : dict or None, optional
            Specifications to run tiled segmentation on the HPC.
            If not provided, will attempt to load from the standard location on disk.
        tiles : list of sis.spot_table.SegmentedSpotTable or None, optional
            The individual tiles that were segmented.
            If not provided, will be generated from spot_table and run_spec.
        detect_z_planes : float or None, optional
            If provided, limit the z-planes to those that contain at least *detect_z_planes* fraction of spots.
            Should be the same as the value used in the segmentation method.
        overwrite : bool, optional
            Whether to overwrite the cell ids file.
        
        Returns
        -------
        (cell_ids, merge_results, skipped) : tuple[numpy.ndarray, list[dict], list[int]]
            - The array of cell_ids corresponding to each spot.
            - Information about merge conflicts collected during tile merging.
            - Indices of tiles skipped during segmentation.
            
        Raises
        ------
        RuntimeError
            If no tiles generated output files, indicating an issue with the segmentation process.
        """
        if run_spec is None:
            run_spec = self.load_run_spec(self.seg_run_spec_path)

        print('Merging tiles...')
        truncated_meta = {
                'seg_method': str(self.seg_method),
                'seg_opts': self.seg_opts,
                'polygon_opts': self.polygon_opts
                }

        # Merging updates the spot table cell_ids in place so we define empty SegmentedSpotTable to dump result into
        self.seg_spot_table = SegmentedSpotTable(
                spot_table=self.raw_spot_table, 
                cell_ids=np.empty(len(self.raw_spot_table), dtype=int),
                seg_metadata=truncated_meta,
                )

        if tiles is None:
            tiles = []
            for tile_spec in run_spec.values():
                # Recreate each tile from spot_table and run_spec
                tile_rgn = tile_spec[2]['subregion']
                tile = self.raw_spot_table.get_subregion(xlim = tile_rgn[0], ylim = tile_rgn[1])
                tiles.append(tile)

        merge_results = []
        skipped = []
        for i, (tile_spec, tile) in enumerate(tqdm(zip(run_spec.values(), tiles))):
            cell_id_file = tile_spec[2]['cell_id_file']
            if not os.path.exists(cell_id_file):
                print(f"Skipping tile {i} : no cell ID file generated")
                skipped.append(i)
                continue
            
            if detect_z_planes: 
                # If we detected z-planes to segment on, we must account for that now as cell ids will only be present for some planes
                tile = SegmentedSpotTable(tile, np.zeros(len(tile), dtype=int))
                z_plane_mask = tile.z_plane_mask(tile.detect_z_planes(float_cut=detect_z_planes))
                tile.cell_ids[z_plane_mask] = np.load(cell_id_file)
            else:
                tile_cids = np.load(cell_id_file)
                tile = SegmentedSpotTable(tile, tile_cids)

            tiles[i] = tile
        # padding removes cells which are close to edge and may be poorly segmented
        # padding is set to half the user defined cell seize
        merge_results = self.seg_spot_table.set_cell_ids_from_tiles(tiles, padding=self.seg_opts['cell_dia'] / 2)

        cell_ids = self.seg_spot_table.cell_ids

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save cell_ids
        self.save_cell_ids(cell_ids, overwrite)

        return cell_ids, merge_results, skipped

    def get_polygon_run_spec(self, overwrite: bool=False):
        """Generates a run spec for running cell polygon jobs on the HPC.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the run spec if it exists. Default False.
        
        Returns
        -------
        dict
            The polygon run specifications.
            
        Raises
        ------
        FileExistsError
            If the polygon subset file already exists and overwrite is False.
        """
        self.polygon_subsets_path.mkdir(exist_ok=True)

        # Polygon jobs are split by cells--not area--so we need to get the number of cells
        num_cells = len(self.seg_spot_table.unique_cell_ids)

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
            np.save(self.polygon_subsets_path / f'cell_id_subset_{i}.npy', self.seg_spot_table.unique_cell_ids[start_idx:end_idx])
            
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
        run_spec : dict or None, optional
            The run specification for calculating polygons on the HPC. 
            If not provided, will attempt to load from the standard location on disk.
        overwrite : bool, optional
            Whether to overwrite the polygon run spec if it exists in the output directory. Default False.
            
        Returns
        -------
        (dict, list) : tuple[dict, list[int]]
            - The cell polygons after merging.
            - Indices of cell polygons that were skipped.
            
        Raises
        ------
        RuntimeError
            If all no polygon output files were generated, indicating an issue with the process.
        FileExistsError
            If the polygons file already exists and overwrite is False.
        """
        if run_spec is None:
            run_spec = self.load_run_spec(self.polygon_run_spec_path)

        print('Merging cell polygons...')
        skipped = []
        for i, area_spec in enumerate(tqdm(run_spec.values())):
            result_file = area_spec[2]['result_file']
            cell_subset_file = area_spec[2]['cell_subset_file']
            if not os.path.exists(result_file):
                print(f"Skipping tile {i} : no result file generated")
                skipped.append(i)
                continue
            # The reset_cache=False is important to allow reading in the various cell subsets without overwriting
            self.seg_spot_table.load_cell_polygons(result_file, cell_ids=cell_subset_file, reset_cache=False, disable_tqdm=True)

        if len(run_spec) == len(skipped):
            raise RuntimeError('All tiles were skipped, check error logs.')

        if len(skipped) > 0:
            print('Warning: Some tiles were skipped.')

        # save polygons
        if not overwrite and self.polygon_final_path.exists():
            raise FileExistsError('cell polygons already saved and overwriting is not enabled.')
        self.seg_spot_table.save_cell_polygons(self.output_dir / f'cell_polygons.{self.polygon_opts["save_file_extension"]}')

        return self.seg_spot_table.cell_polygons, skipped
    
    
    def create_cell_by_gene(self, x_format: str, x_dtype: str='uint16', additional_obs: dict|None=None, prefix: str='', suffix: str='', overwrite: bool=False):
        """Create and save a cell by gene AnnData object from the attached spot table.
        
        Parameters
        ----------
        x_format : str
            Desired format for the cell by gene anndata X. Options: 'dense' or
            'sparse'.
        x_dtype : str, optional
            The data type of the matrix.
        additional_obs : dict or None, optional
            Additional columns to add to the anndata.obs DataFrame.
            Keys are column names and values are arrays of the same length as the number of cells.
        prefix : str, optional
            The string to prepend to all cell labels
        suffix : str, optional
            The string to append to all cell labels
        overwrite : bool, optional
            Whether to allow overwriting of output files. Default False.
        
        Returns
        -------
        cell_by_gene : anndata.AnnData
            The cell by gene table.
        """
        self.seg_spot_table.generate_cell_labels(prefix=prefix, suffix=suffix)
        cell_by_gene = self.seg_spot_table.cell_by_gene_anndata(x_format=x_format, x_dtype=x_dtype, additional_obs=additional_obs)
        self.save_cbg(cell_by_gene, overwrite)

        return cell_by_gene
        
    def clean_up(self, mode="all_ints"):
        """Clean up intermediate files after segmentation and polygon generation is complete.

        Parameters
        ----------
        mode : str, optional 
            Can be 'all_ints', 'seg_ints', 'polygon_ints', or 'none' depending on desired clean up. Defaults to 'all_ints'.
            
        Raises
        ------
        ValueError
            If an invalid clean up mode is specified. (i.e. not 'all_ints', 'seg_ints', 'polygon_ints', or 'none').
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
        """Alternate constructor to load from a SpatialDataset
        
        Parameters
        ----------
        sp_dataset : sis.spatial_dataset.SpatialDataset
            The SpatialDataset to load from.
        output_dir : Path
            The directory to save output files to.
        subrgn : tuple
            The subregion for segmentation.
        seg_method : class
            The segmentation method class.
        seg_opts : dict
            The options for the segmentation method.
        polygon_opts : dict or None, optional
            The options for calculating cell polygons. Default None.
        seg_hpc_opts : dict or None, optional
            The options for running segmentation on the HPC. Default None.
        polygon_hpc_opts : dict or None, optional
            The options for running cell polygon calculation on the HPC. Default None.
        hpc_opts : dict or None, optional
            The options for running on the HPC. Default None.
        """
        image_path = sp_dataset.images_path
        csv_file = sp_dataset.detected_transcripts_file
        cache_file = sp_dataset.detected_transcripts_cache

        return cls(csv_file, image_path, output_dir, cache_file, subrgn, seg_method, seg_opts, hpc_opts, polygon_opts, seg_hpc_opts, polygon_hpc_opts, hpc_opts)

    @classmethod
    def from_json(cls, json_file):
        """Alternate constructor to load a SegmentationPipeline from a json file
        
        Parameters
        ----------
        json_file : Path
            The path to the json file.
            
        Raises
        ------
        NotImplementedError
            If the segmentation method specified in the json file does not exist
        """
        with open(json_file, 'r') as f:
            config = json.load(f)
            
        # Cannot save the class in the json so we need to convert it back from string to the class
        seg_method_name = config['seg_method'].rpartition('.')[-1]
        if seg_method_name == 'CellposeSegmentationMethod':
            config['seg_method'] = sis.segmentation.CellposeSegmentationMethod
        else:
            raise NotImplementedError(f'Segmentation method {seg_method_name} not implemented.')
        
        # Cannot save tuple in the json so we need to convert back from list
        if isinstance(config['subrgn'], list):
            config['subrgn'] = tuple([tuple(l) for l in config['subrgn']])

        return cls(**config)

    def rerun_failed_jobs(self, job_type: str, jobs: SlurmJobArray, run_spec: dict, mem: str|None=None, time: str|None=None, max_attempts: int=5):
        """This function takes jobs and the run_spec that submitted them and resubmits any failed jobs.
        It continues to resubmit until all jobs are completed properly or the maximum number of attempts is reached

        Parameters
        ----------
        job_type : str
            The type of jobs to rerun. Set to 'segmentation' to run tiled segmentation or 'cell_polygons' to calculate cell polygons.
        jobs : sis.hpc.SlurmJobArray
            A SlurmJobArray instance containing jobs to check for failures
        run_spec : dict
            The run_spec that was used to submit the jobs previously
        mem : str or None, optional
            The amount of memory that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's memory
        time : str or None, optional
            The length of time that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's time
        max_attempts : int, optional
            The maximum number of times to attempt to rerun the failed jobs. Default 5
            
        Returns
        -------
        jobs : SlurmJobArray
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
        """This function takes a sis.hpc.SlurmJobArray and identifies which jobs—-if any--failed and how they failed
        
        Parameters
        ----------
        jobs: sis.hpc.SlurmJobArray
            A SlurmJobArray instance containing jobs to check for failures
        
        Returns
        ----------
        (None, None) : tuple[None, None]
            Indicates that all jobs completed successfully.
        (to_rerun, failure_types) : tuple[list[int], dict]
            If some jobs failed, returns a tuple containing:
            - A list of the indices of the failed jobs in the inputted SlurmJobArray
            - A dictionary with keys as types of failues and bools representing if that failure occured in the inputted SlurmJobArray
                
        Raises
        ------
        ValueError
            If the job state is not one of OUT_OF_MEMORY, TIMEOUT, or CANCELLED.
            This indicates that the job failed for an unknown reason and cannot be automatically rerun.
        """
        job_state_dict = jobs.state()
        # Only supported failure types, as others are not automatically addressable
        failure_types= {"OUT_OF_MEMORY": False, "TIMEOUT": False, "CANCELLED": False}
        failed_jobs_indices = []
        for job_index, job_state in enumerate(job_state_dict.values()):
            if job_state.state == "COMPLETED":
                continue
            elif job_state.state in failure_types.keys():
                failed_jobs_indices.append(job_index)
                failure_types[job_state.state] = True # want to tell the user why their jobs failed
            else:
                raise ValueError(f"Could not automatically rerun jobs. Job state must be one of OUT_OF_MEMORY, TIMEOUT, or CANCELLED. Job state was {job_state.state}")
        
        if len(failed_jobs_indices) == 0:
            return None, None# No jobs to rerun
        else:
            return failed_jobs_indices, failure_types
        
    def resubmit_failed_jobs(self, job_type: str, indices_to_rerun: list[int], failure_types: dict, run_spec: dict, mem: str|None=None, time: str|None=None):
        """This function resubmits failed jobs.
        
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
        mem: str or None, optional
            The amount of memory that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's memory
        time: str or None, optional
            The length of time that should be allocated to the job reruns
            If the amount is less than the amount used to submit the failed job or left unspecified, it defaults to doubling the previous run's time
        
        Returns
        -------
        sis.hpc.SlurmJobArray
            A sis.hpc.SlurmJobArray containing all the resubmitted jobs
            
        Raises
        ------
        ValueError
            If the job type is not one of 'segmentation' or 'cell_polygons'.
        """
        if 'segmentation' in job_type:
            hpc_opts = self.seg_hpc_opts
        elif 'cell_polygons' in job_type:
            hpc_opts = self.polygon_hpc_opts
        else:
            raise ValueError('Invalid job type.')
        
        # Query the run_specs of failed jobs so we can resubmit them
        # We can only fix HPC issues automatically, so the run_spec will stay the same
        new_run_spec = {}
        for job_array_index, failed_tile_index in enumerate(indices_to_rerun):
            new_run_spec[job_array_index] = run_spec[failed_tile_index]
        
        # If job failed due to memory constraints we adjust the hpc options accordingly
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
        
        # If job failed due to time constraints, we adjust hpc options accordingly
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
        """Replace jobs in a SlurmJobArray with new jobs.
        
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
        jobs : sis.hpc.SlurmJobArray
            The updated SlurmJobArray
        """
        for new_jobs_idx, (jobs_idx, job) in enumerate(zip(indices_to_replace, new_jobs.jobs)):
            jobs.jobs[jobs_idx] = SlurmJob(job.args, job.sbatch_output, job.job_file, job.host, array_id=new_jobs_idx, job_array=jobs)
        return jobs


class MerscopeSegmentationPipeline(SegmentationPipeline):
    """Class for running segmentation on Merscope data.
    
    Attributes
    ----------
    See sis.segmentation.SegmentationPipeline for inherited attributes.
    """
    def __init__(self, dt_file: Path|str, image_path: Path|str, output_dir: Path|str, dt_cache: Path|str|None, subrgn: str|tuple, seg_method: SegmentationMethod, seg_opts: dict, polygon_opts: dict|None=None, seg_hpc_opts: dict|None=None, polygon_hpc_opts: dict|None=None, hpc_opts: dict|None=None):
        """
        Parameters
        ----------
        dt_file : str or Path
            Path to the detected transcripts file.
        image_path : str or Path
            Path to the images.
        output_dir : str or Path
            Where to save output files.
        dt_cache : str or Path or None
            Path to the detected transcripts cache file. Used for faster loading.
        subrgn : str or tuple
            The subregion to segment. Set to a string, e.g. 'DAPI', to segment the 
            full region bounded by the associated image channel. To segment a 
            smaller region, set to a tuple corresponding to a bounding box.
        seg_method : SegmentationMethod
            The segmentation method to use. Must be found in sis.segmentation.
        seg_opts : dict
            Options to pass to seg_method.
        polygon_opts : dict or None, optional
            Options to pass to for cell polygon generation. Currently supports save_file_extension and alpha_inv_coeff.
            Default is None, which sets save_file_extension to 'geojson' and alpha_inv_coeff to 4/3.    
        seg_hpc_opts : dict, None or None, optional
            Options to use for segmenting tiles on the hpc. Default is None
        polygon_hpc_opts : dict, None or None, optional
            Options to use for calculating cell polygons on the hpc. Default is None
        hpc_opts : dict, None or None, optional
            Options to use for both segmenting tiles and calculating cell polygons on the hpc (can be used in place of submitting both seg_hpc_opts and polygon_hpc_opts).
            Default is None
        """
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, polygon_opts, seg_hpc_opts=seg_hpc_opts, polygon_hpc_opts=polygon_hpc_opts, hpc_opts=hpc_opts)

    def get_load_func(self):
        """Get the function to load a spot table.
        """
        return SpotTable.load_merscope

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc).
        """
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
    """Class for running segmentation on StereoSeq data.
    
    Attributes
    ----------
    See sis.segmentation.SegmentationPipeline for inherited attributes.
    """
    def __init__(self, dt_file: Path|str, image_path: Path|str, output_dir: Path|str, dt_cache: Path|str|None, subrgn: str|tuple, seg_method: SegmentationMethod, seg_opts: dict, polygon_opts: dict|None=None, seg_hpc_opts: dict|None=None, polygon_hpc_opts: dict|None=None, hpc_opts: dict|None=None):
        """
        Parameters
        ----------
        dt_file : str or Path
            Path to the detected transcripts file.
        image_path : str or Path
            Path to the images.
        output_dir : str or Path
            Where to save output files.
        dt_cache : str or Path or None
            Path to the detected transcripts cache file. Used for faster loading.
        subrgn : str or tuple
            The subregion to segment. Set to a string, e.g. 'DAPI', to segment the 
            full region bounded by the associated image channel. To segment a 
            smaller region, set to a tuple corresponding to a bounding box.
        seg_method : SegmentationMethod
            The segmentation method to use. Must be found in sis.segmentation.
        seg_opts : dict
            Options to pass to seg_method.
        polygon_opts : dict or None, optional
            Options to pass to for cell polygon generation. Currently supports save_file_extension and alpha_inv_coeff.
            Default is None, which sets save_file_extension to 'geojson' and alpha_inv_coeff to 4/3.    
        seg_hpc_opts : dict or None, optional
            Options to use for segmenting tiles on the hpc. Default is None
        polygon_hpc_opts : dict or None, optional
            Options to use for calculating cell polygons on the hpc. Default is None
        hpc_opts : dict or None, optional
            Options to use for both segmenting tiles and calculating cell polygons on the hpc (can be used in place of submitting both seg_hpc_opts and polygon_hpc_opts).
            Default is None
        """
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

class XeniumSegmentationPipeline(SegmentationPipeline):
    """Class for running segmentation on Xenium data.
    
    Attributes
    ----------
    See sis.segmentation.SegmentationPipeline for inherited attributes.
    z_depth : float
        The depth of each z-plane. Used to assign z-locations to image planes.
    cache_image : bool
        Whether to cache images after retrieval. Default True.
    """
    def __init__(self, dt_file: Path|str, image_path: Path|str, output_dir: Path|str, dt_cache: Path|str|None, subrgn: str|tuple, seg_method: SegmentationMethod, seg_opts: dict, polygon_opts: dict|None=None, seg_hpc_opts: dict|None=None, polygon_hpc_opts: dict|None=None, hpc_opts: dict|None=None, cache_image: bool=True):
        """
        Parameters
        ----------
        dt_file : str or Path
            Path to the detected transcripts file.
        image_path : str or Path
            Path to the images.
        output_dir : str or Path
            Where to save output files.
        dt_cache : str or Path or None
            Path to the detected transcripts cache file. Used for faster loading.
        subrgn : str or tuple
            The subregion to segment. Set to a string, e.g. 'DAPI', to segment the 
            full region bounded by the associated image channel. To segment a 
            smaller region, set to a tuple corresponding to a bounding box.
        seg_method : SegmentationMethod
            The segmentation method to use. Must be found in sis.segmentation.
        seg_opts : dict
            Options to pass to seg_method.
        polygon_opts : dict or None, optional
            Options to pass to for cell polygon generation. Currently supports save_file_extension and alpha_inv_coeff.
            Default is None, which sets save_file_extension to 'geojson' and alpha_inv_coeff to 4/3.    
        seg_hpc_opts : dict or None, optional
            Options to use for segmenting tiles on the hpc. Default is None
        polygon_hpc_opts : dict or None, optional
            Options to use for calculating cell polygons on the hpc. Default is None
        hpc_opts : dict or None, optional
            Options to use for both segmenting tiles and calculating cell polygons on the hpc (can be used in place of submitting both seg_hpc_opts and polygon_hpc_opts).
            Default is None
        cache_image : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not.
            The trade off is speed vs memory.
            
        Raises
        ------
        ValueError
            If 'z_plane_thickness' is not provided in seg_opts. This is required to match z coordinates to image planes.
        """
        super().__init__(dt_file, image_path, output_dir, dt_cache, subrgn, seg_method, seg_opts, polygon_opts, seg_hpc_opts=seg_hpc_opts, polygon_hpc_opts=polygon_hpc_opts, hpc_opts=hpc_opts)

        # Couple extra variables for Xenium segmentation
        if 'z_plane_thickness' not in seg_opts:
            raise ValueError('z_plane_thickness required in seg_opts for matching z coordinates to image planes')
        self.z_depth = seg_opts['z_plane_thickness'] # This is used for binning z-locations to image planes
        self.cache_image = cache_image

    def get_load_func(self):
        """Get the function to load a spot table.
        """
        return SpotTable.load_xenium

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc).
        """
        load_args = {
                'image_path': self.image_path,
                'transcript_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
        }
        for k, v in load_args.items():
            if isinstance(v, Path):
                load_args[k] = v.as_posix()

        load_args['max_rows'] = None
        load_args['z_depth'] = self.z_depth
        load_args['cache_image'] = self.cache_image

        return load_args
