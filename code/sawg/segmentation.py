from __future__ import annotations
import os, tempfile, pickle, traceback
import scipy.ndimage, scipy.interpolate
import numpy as np
from .spot_table import SpotTable
from .image import Image, ImageBase, ImageTransform


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
        }
        cp_opts.update(self.options['cellpose_options'])
        cp_opts.setdefault('anisotropy', self.options['z_plane_thickness'] / self.options['px_size'])
        cp_opts.setdefault('diameter', self.options['cell_dia'] / self.options['px_size'])
        
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
        model = cellpose.models.Cellpose(model_type=self.options['cellpose_model'], gpu=gpu)

        # run segmentation
        masks, flows, styles, diams = model.eval(image_data, **cp_opts)

        dilate = self.options.get('dilate', 0)
        if dilate != 0:
            masks = dilate_labels(masks, radius=dilate/self.options['px_size'])

        # return result object
        result = CellposeSegmentationResult(
            method=self, 
            input_spot_table=spot_table,
            cellpose_output={
                'masks': masks, 
                'flows': flows, 
                'styles': styles, 
                'diams': diams,
            },
            image_transform=first_image.transform,
        )
            
        return result

    def _read_image_spec(self, img_spec, spot_table, px_size, images):
        """Return an Image to be used in segmentation based on img_spec:
        
        - An Image instance is returned as-is
        - "total_mrna" returns an image generated from spot density
        - Any other string returns an image channel attached to the spot table
        - {'channel': channel, 'frame': int} can be used to select a single frame
        - {'channel': 'total_mrna', 'gauss_kernel': (1, 3, 3), 'median_kernel': (2, 10, 10)} can be ued to configure total mrna image generation
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

    def get_total_mrna_image(self, spot_table, image_shape:tuple, image_transform:ImageTransform, gauss_kernel=(1, 3, 3), median_kernel=(2, 10, 10)):
            
        density_img = np.zeros(image_shape[:3], dtype='float32')

        spot_px = self.map_spots_to_img_px(spot_table, image_transform=image_transform, image_shape=image_shape)
        n_planes = density_img.shape[0]
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

class MetaSegmentationMethod(SegmentationMethod):
    """Implements 2D or 3D Meta SegmentAnything segmentation on SpotTable

    Will automatically segment from images attached to the SpotTable or
    generate an image from total mRNA.
    
    options = {
        'region': ((xmin, xmax), (ymin, ymax)),  # or None for whole table
        'images': {
            'nuclei': 'DAPI',
            'cyto': 'total_mrna',
        },
        'meta_options': {
            points_per_side: 32,
            points_per_batch: 64,
            pred_iou_thresh: 0.88,
            stability_score_thresh: 0.95,
            stability_score_offset: 1.0,
            box_nms_thresh: 0.7,
            crop_n_layers: 0,
            crop_nms_thresh: 0.7,
            crop_overlap_ratio: 512 / 1500,
            crop_n_points_downscale_factor: 1,
            point_grids: None,
            min_mask_region_area: 0,
            output_mode: "binary_mask",
        },
    }
    """
    def __init__(self, options):
        super().__init__(options)

    def run(self, spot_table):
        spot_table = self._get_spot_table(spot_table)

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        sam_checkpoint = self.options['meta_options']['sam_checkpoint']#"sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)

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

        if len(self.images) == 1:
            image_type = list(self.images.keys())[0]
            image_data = np.stack((self.images[image_type].get_data().mean(axis=0), self.images[image_type].get_data().mean(axis=0), self.images[image_type].get_data().mean(axis=0)), axis=-1)
        elif len(self.images) == 2:
            min_x = min(self.images['cyto'].shape[1], self.images['nuclei'].shape[1])
            min_y = min(self.images['cyto'].shape[2], self.images['nuclei'].shape[2])
            image_data = np.stack((self.images['cyto'].get_data().mean(axis=0)[:min_x, :min_y], self.images['nuclei'].get_data().mean(axis=0)[:min_x, :min_y], np.zeros((min_x, min_y))), axis=-1)
        else:
            raise ValueError('Unexpected number of image layers')

        image_data = (np.ma.divide(image_data - image_data.min(axis=(0,1)), image_data.max(axis=(0,1)) - image_data.min(axis=(0,1))).filled(0) * 255).astype(np.uint8) # Must transform to int for SAM to work
        masks = mask_generator.generate(image_data)

        # Deal with massive spike for full image mask
        area_check_list = sorted([x['area'] for x in masks])[::-1]
        if area_check_list[0] >= 25 * area_check_list[1]:
            for i, x in enumerate(masks):
                if x['area'] == area_check_list[0]:
                    full_image_idx = i
            masks.pop(full_image_idx)

        result = MetaSegmentationResult(
            method=self, 
            input_spot_table=spot_table,
            meta_output={
                'masks': masks
            },
            image_transform=list(images.values())[0].transform,
        )

        return result


    def _read_image_spec(self, img_spec, spot_table, px_size, images):
        """Return an Image to be used in segmentation based on img_spec:
        
        - An Image instance is returned as-is
        - "total_mrna" returns an image generated from spot density
        - Any other string returns an image channel attached to the spot table
        - {'channel': channel, 'frame': int} can be used to select a single frame
        - {'channel': 'total_mrna', 'gauss_kernel': (1, 3, 3), 'median_kernel': (2, 10, 10)} can be ued to configure total mrna image generation
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


    def get_total_mrna_image(self, spot_table, image_shape:tuple, image_transform:ImageTransform, gauss_kernel=(1, 3, 3), median_kernel=(2, 10, 10)):
            
        density_img = np.zeros(image_shape[:3], dtype='float32')

        spot_px = self.map_spots_to_img_px(spot_table, image_transform=image_transform, image_shape=image_shape)
        n_planes = density_img.shape[0]
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


class MetaSegmentationResult(SegmentationResult):
    def __init__(self, method:SegmentationMethod, input_spot_table:SpotTable, meta_output:dict, image_transform:ImageTransform):
        super().__init__(method, input_spot_table)
        self.meta_output = meta_output
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
            if self.meta_output['masks'][0]['segmentation'].ndim == 2:
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
        masks = self.meta_output['masks']
        # Pull out the actual image data
        img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]), dtype=np.int64)
        for i, mask in enumerate(masks):
            img[mask['segmentation']] = i

        if img.ndim == 2:
            # add frame and channel axes back
            img = img[np.newaxis, ..., np.newaxis]
        else:
            # add channel axis back
            img = img[..., np.newaxis]
        return Image(data=img, transform=self.image_transform, channels=['Meta Mask'], name=None)
        
    def show(self, ax):
        """Display segmentation result in matplotlib axes
        """
        
        
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
