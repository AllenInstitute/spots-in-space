from __future__ import annotations
import os, tempfile, pickle
import numpy as np
from .spot_table import SpotTable
from .image import Image, ImageBase, ImageTransform


def run_segmentation(load_func, load_args:dict, subregion:dict|None, method_class, method_args:dict, output_file:str):
    """Load a spot table, run segmentation (possibly on a subregion), and save the SegmentationResult.
    """
    spot_table = load_func(**load_args)
    if subregion is not None:
        spot_table = spot_table.get_subregion(**subregion)
    seg = method_class(**method_args)
    result = seg.run(spot_table)
    result.save(output_file)


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
        }
    }
    """
    
    def __init__(self, options):
        super().__init__(options)
        
    def run(self, spot_table):
        import cellpose
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
            images['nuclei'] = self._read_image_spec(img_spec, spot_table)
        if 'cyto' in self.options['images']:
            img_spec = self.options['images']['cyto']
            img = self._read_image_spec(img_spec, spot_table, nuclei_img=images.get('nuclei', None))
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
            
        # initialize cellpose model
        cp_opts['channels'] = channels
        model = cellpose.models.Cellpose(model_type=self.options['cellpose_model'], gpu=self.options.get('cellpose_gpu', False))

        # run segmentation
        masks, flows, styles, diams = model.eval(image_data, **cp_opts)
        
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

    def _read_image_spec(self, img_spec, spot_table, nuclei_img=None):
        """Return an Image to be used in segmentation based on img_spec:
        
        - An Image instance is returned as-is
        - "total_mrna" returns an image generated from spot density
        - Any other string returns an image channel attached to the spot table
        """
        # optionally, cyto image may be generated from spot table total mrna
        if img_spec == 'total_mrna':
            return self.get_total_mrna_image(spot_table, nuclei_img=nuclei_img)
        elif isinstance(img_spec, str):
            return spot_table.get_image(channel=img_spec)
        elif isinstance(img_spec, ImageBase):
            return img_spec
        else:
            raise TypeError("Bad image spec:", img_spec)

    def get_total_mrna_image(self, spot_table, nuclei_img, gauss_kernel=(1, 3, 3), median_kernel=(2, 10, 10)):
        if nuclei_img is None:
            raise NotImplementedError()  # we just need to make up a reasonable image resolution instead
            
        density_img = np.zeros(nuclei_img.shape[:3], dtype='float32')

        spot_px = self.map_spots_to_img_px(spot_table, nuclei_img)
        n_planes = density_img.shape[0]
        for i in range(n_planes):
            z_mask = spot_px[..., 0] == i
            x = spot_px[z_mask, 1]
            y = spot_px[z_mask, 2]
            bins = [
                # is this correct? test on rectangular tile
                np.arange(nuclei_img.shape[1]+1),
                np.arange(nuclei_img.shape[2]+1),
            ]
            density_img[i] = np.histogram2d(x, y, bins=bins)[0]

        import scipy.ndimage
        # very sensitive to these parameters :/
        density_img = scipy.ndimage.gaussian_filter(density_img, gauss_kernel)
        density_img = scipy.ndimage.median_filter(density_img, median_kernel)
        
        # todo: use a global normalization range
        
        density_img = density_img * (nuclei_img.get_data().max() / density_img.max())
        
        return Image(density_img[..., np.newaxis], transform=nuclei_img.transform, channels=['Total mRNA'], name=None)

    def map_spots_to_img_px(self, spot_table:SpotTable, image:Image):
        """Map spot table (x, y) positions to image (row, col)
        """
        spot_xy = spot_table.pos[:, :2]
        spot_px_rc = np.floor(image.transform.map_to_pixels(spot_xy)).astype(int)

        # for this dataset, z values are already integer index instead of um
        spot_px_z = spot_table.z.astype(int)

        # some spots may be a little past the edge of the image; 
        # just clip these as they'll be discarded when tiles are merged anyway
        spot_px_rc[:, 0] = np.clip(spot_px_rc[:, 0], 0, image.shape[1]-1)
        spot_px_rc[:, 1] = np.clip(spot_px_rc[:, 1], 0, image.shape[2]-1)
        
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
