import numpy as np
import rasterio
from rasterio.windows import Window


class ImageBase:
    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        """
        raise NotImplementedError()

    def get_data(self, channel=None):
        raise NotImplementedError()

    def get_channel(self, channel):
        assert channel in self.channels
        return ImageView(self, channels=[channel])
        
    def get_subregion(self, region):
        """Return a view of this image limited to the region [(xmin, ymin), (xmax, ymax)]
        """
        corners = np.array(region)
        tl, br = self.transform.map_to_pixels(corners)
        return self[:, tl[0]:br[0], tl[1]:br[1]]

    def __getitem__(self, item):
        return ImageView(self, slices=item)

    def get_z_index(self, z):
        z_len = self.shape[0]
        assert z < z_len
        return self[z:z+1, ...]

        
class Image(ImageBase):
    def __init__(self, file: str, transform: np.ndarray, axes: list, channels: list, name: str|None):
        """Represents a single image, carrying metadata about:
        - The file containing image data
        - The transform that maps from pixel coordinates to spot table coordinates
        - Which axes are which
        - What is represented by each channel

        Image data are lazy-loaded so that we can handle subregions without loading the entire image
        
        Parameters
        ----------
        file : str
            Path to image file
        transform : ndarray
            3D transformation matrix relating (frame, row, col) image pixel coordinates to (x, y, z) spot coordinates.
        axes : list
            List of axis names giving order of axes in *file*; options are 'frame', 'row', 'col', 'channel'
        channels : list
            List of names given to each channel (e.g.: 'dapi')
        name : str|None
            Optional unique identifier for this image
        """
        super().__init__(self)
        self.file = file
        self.transform = transform
        self.axes = axes
        self.channels = channels
        self.name = name
        self._shape = None

    @classmethod
    def load_merscope(cls, image_file, transform_file, channel, name=None):
        um_to_px = np.loadtxt(transform_file)
        tr = ImageTransform(um_to_px)
        return Image(file=image_file, transform=tr, axes=['frame', 'row', 'col', 'channel'], channels=[channel], name=name)

    @property
    def shape(self):
        if self._shape is None:
            with rasterio.open(self.file) as src:
                # todo: support for multiple planes per file?
                self._shape = (1, src.meta['height'], src.meta['width'], src.meta['count'])
        return self._shape

    def get_data(self, channel=None):
        """Return array of image data.

        If the image has multiple channels, then the name of the channel to return must be given.
        """
        index = self._get_channel_index(channel)
        with rasterio.open(self.file) as src:
            return src.read(index)

    def get_sub_data(self, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion

        Parameters
        ----------
        rows : tuple
            (first_row, last_row+1)
        cols : tuple
            (first_col, last_col+1)
        channel : str | None
            Name of channel to return data from
        """
        index = self._get_channel_index(channel)
        win = Window(rows[0], cols[0], rows[1]-rows[0], cols[1]-cols[0])
        with rasterio.open(self.file) as src:
            return src.read(index, window=win)

    def _get_channel_index(self, channel):
        # rasterio indexes channels starting at 1
        if channel is None and len(self.channels) == 1:
            return 1
        else:
            return self.channels.index(channel) + 1


"""
import rasterio
f = '/allen/programs/celltypes/workgroups/rnaseqanalysis/NHP_spatial/MERSCOPE/macaque/1191380492/images/mosaic_DAPI_z3.tif'
src = rasterio.open(f)
from rasterio.windows import Window
win = Window(25600, 25600, 128, 128)
sub = src.read(window=win)
"""
class ImageTransform:
    def __init__(self, matrix):
        self.matrix = matrix
        assert matrix.shape == (2, 3)

    def map_to_pixels(self, points):
        """Map x,y positions to image pixels. 
        
        Points must be an array of shape (N, 2).
        """
        return (self.matrix[:2, :2] @ points.T + self.matrix[:2, 2:]).astype(int).T



class ImageStack(ImageBase):
    """A stack of Image z-planes
    """
    def __init__(self, images):
        super().__init__(self)
        self.images = images


class ImageView(ImageBase):
    """Represents a subset of data from an Image (a rectangular subregion or subset of channels)
    """
    def __init__(self, image, slices=None, channels=None):
        super().__init__(self)
        for ch in channels:
            assert ch in image.channels
        self.image = image
        self.view_slices = slices
        self.view_channels = channels
        
    @property
    def channels(self):
        return self.view_channels

    def get_data(self, channel=None):
        rows = self.view_slices[1].start, self.view_slices[1].stop
        cols = self.view_slices[2].start, self.view_slices[2].stop
        return self.image.get_sub_data(rows, cols, channel=channel)

    def get_sub_data(self, rows, cols, channel=None):
        rowstart = self.view_slices[1].start
        colstart = self.view_slices[2].start
        rows = (rows[0] + rowstart, rows[1] + rowstart)
        cols = (cols[0] + colstart, cols[1] + colstart)
        return self.image.get_sub_data(rows, cols, channel=channel)

