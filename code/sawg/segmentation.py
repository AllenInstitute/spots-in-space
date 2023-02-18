from .spot_table import SpotTable
from .image import Image, ImageBase, ImageTransform
import numpy as np


class SegmentationResult:
    """Represents a segmentation of SpotTable data--method, options, results
    """
    def __init__(self, method, spot_table):
        self.method = method
        self.input_spot_table = spot_table

    @property
    def cell_ids(self):
        """Array of segmented cell IDs for each spot in the table
        """
        raise NotImplementedError()        

    @property
    def spot_table(self):
        return self.input_spot_table.copy(cell_ids=self.cell_ids)


class SegmentationMethod:
    """Base class defining segmentation methods
    """
    def __init__(self, options):
        self.options = options
        
    def run(self, spot_table):
        """Run segmentation on spot_table and return a Segmentation object.
        """
        raise NotImplementedError()
        
    def get_spot_table(self, spot_table):
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
    """Implements 3D cellpose segmentation on SpotTable
    
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
        spot_table = self.get_spot_table(spot_table)
        
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
            spot_table=spot_table,
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
    def __init__(self, method:SegmentationMethod, spot_table:SpotTable, cellpose_output:dict, image_transform:ImageTransform):
        super().__init__(method, spot_table)
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


def run_baysor(baysor_bin, input_file, output_file, scale=5):
    os.system(f'{baysor_bin} run {input_file} -o {output_file} -s {scale} --no-ncv-estimation')

