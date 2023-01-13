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
        if len(self.channels) == 1:
            return self
        return ImageView(self, channels=[channel])
        
    def get_subregion(self, region):
        """Return a view of this image limited to the region [(xmin, xmax), (ymin, ymax)]
        """
        corners = np.array(region).T
        tl, br = self.transform.map_to_pixels(corners).astype(int)
        return self[:, tl[0]:br[0], tl[1]:br[1]]

    def __getitem__(self, item):
        return ImageView(self, slices=item)

    def get_z_index(self, z):
        z_len = self.shape[0]
        assert z < z_len
        return self[z:z+1, ...]

    def show(self, ax, channel=None, **kwds):
        y_inverted = ax.yaxis_inverted()
        data = self.get_data(channel=channel)
        shape = self.shape
        px_corners = np.array([[0, 0], shape[1:3]])
        (left, top), (right, bottom) = self.transform.map_from_pixels(px_corners)
        kwds['extent'] = (left, right, bottom, top)
        ax.imshow(data, **kwds)
        # don't let imshow invert the y axis
        if ax.yaxis_inverted() != y_inverted:
            ax.invert_yaxis()


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
        super().__init__()
        self.file = file
        self.transform = transform
        self.axes = axes
        self.channels = channels
        self.name = name
        self._shape = None

    @classmethod
    def load_merscope(cls, image_file, transform_file, channel, name=None):
        um_to_px = np.loadtxt(transform_file)[:2]
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


class ImageTransform:
    def __init__(self, matrix):
        self.matrix = matrix
        assert matrix.shape == (2, 3)
        self._inverse = None

    def map_to_pixels(self, points):
        """Map (x, y) positions to image pixels (row, col). 
        
        Points must be an array of shape (N, 2).
        """
        return (self.matrix[:2, :2] @ points.T + self.matrix[:2, 2:]).T

    @property
    def inverse_matrix(self):
        if self._inverse is None:
            m3 = np.eye(3)
            m3[:2] = self.matrix
            self._inverse = np.linalg.inv(m3)[:2]
        return self._inverse

    def map_from_pixels(self, pixels):
        """Map (row, col) image pixels to positions (x, y). 
        
        Points must be an array of shape (N, 2).
        """
        return (self.inverse_matrix[:2, :2] @ pixels.T + self.inverse_matrix[:2, 2:]).T

    def translated(self, offset):
        """Return a new transform that is translated by *offset* (where offset is expressed in pixels)
        """
        m = self.matrix.copy()
        m[:, 2] += offset
        return ImageTransform(m)


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
        super().__init__()
        if channels is not None:
            for ch in channels:
                assert ch in image.channels
        self.image = image
        self.view_slices = slices
        self.view_channels = channels

        offset = np.array([0, 0])
        if slices is not None:
            for ax, sl in enumerate(slices[1:3]):
                offset[ax] = sl.start
                # for now, let's not support stepping here
                assert sl.step in (1, None), f"Slice step not supported ({sl})"

        self.transform = image.transform.translated(-offset)

    @property
    def name(self):
        return self.image.name

    @property
    def shape(self):
        shape = list(self.image.shape)
        if self.view_slices is not None:
            for ax, sl in enumerate(self.view_slices):
                inds = sl.indices(shape[ax])
                shape[ax] = (inds[1] - inds[0]) // inds[2]
        if self.view_channels is not None:
            shape[3] = len(self.view_channels)
        return shape

    @property
    def channels(self):
        if self.view_channels is not None:
            return self.view_channels
        else:
            return self.image.channels

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

