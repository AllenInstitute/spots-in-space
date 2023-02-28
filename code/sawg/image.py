from __future__ import annotations
import warnings
import os, glob, re
import numpy as np
from .optional_import import optional_import
rasterio = optional_import('rasterio')


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
        return self.get_pixel_subregion([(tl[0],br[0]), (tl[1],br[1])])

    def get_pixel_subregion(self, region):
        """Return a view of this image limited to the region [:, rowmin:rowmax, colmin:colmax]
        """
        return ImageView(self, rows=region[0], cols=region[1])

    def get_frame(self, frame):
        z_len = self.shape[0]
        assert frame < z_len
        return ImageView(self, frames=(frame, frame+1))

    def show(self, ax=None, frame=None, channel=None, **kwds):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        img = self
        if isinstance(frame, int):
            img = img.get_frame(frame)
            
        data = img.get_data(channel=channel)

        if frame == 'mean':
            data = data.mean(axis=0)
        else:
            data = data[0]

        self._show_image(data, ax, **kwds)

    def _show_image(self, data, ax, **kwds):
        y_inverted = ax.yaxis_inverted()
        shape = self.shape
        px_corners = np.array([[0, 0], shape[1:3]])
        (left, top), (right, bottom) = self.transform.map_from_pixels(px_corners)
        kwds['extent'] = (left, right, bottom, top)
        ax.imshow(data, **kwds)
        # don't let imshow invert the y axis
        if ax.yaxis_inverted() != y_inverted:
            ax.invert_yaxis()


class Image(ImageBase):
    """An Image defined by 4D numpy array (frames, rows, cols, channels).
    
    Carries metadata about pixel transform and channel identity.
    
    """
    def __init__(self, data:np.ndarray, transform:ImageTransform, channels:list, name: str|None=None):
        super().__init__()
        assert data.ndim == 4
        assert isinstance(transform, ImageTransform)
        self.transform = transform
        self.channels = channels
        self.name = name
        self._data = data

    @property
    def shape(self):
        return self._data.shape
        
    def get_data(self, channel=None):
        """Return array of image data.

        If the image has multiple channels, then the name of the channel to return must be given.
        """
        index = self._get_channel_index(channel)
        return self._data[..., index]

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion

        Parameters
        ----------
        frames : tuple
            (first_frame, last_frame)
        rows : tuple
            (first_row, last_row+1)
        cols : tuple
            (first_col, last_col+1)
        channel : str | None
            Name of channel to return data from
        """
        chan = self.get_data(channel)
        return chan[frames[0]:frames[1], rows[0]:rows[1], cols[0]:cols[1]]

    def _get_channel_index(self, channel):
        if channel is not None:
            return self.channels.index(channel)
        else:
            assert self.shape[3] == 1, "Must specify channel to return"
            return 0


class ImageFile(ImageBase):
    def __init__(self, file: str, transform:ImageTransform, axes: list|None, channels: list, name: str|None):
        """Represents a single image stored on disk, carrying metadata about:
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
            ImageTransform relating (row, col) image pixel coordinates to (x, y) spot coordinates.
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
        # transpose rows so we map to (row,col) instead of (col,row)
        um_to_px = um_to_px[::-1]
        tr = ImageTransform(um_to_px)
        return ImageFile(file=image_file, transform=tr, axes=['frame', 'row', 'col', 'channel'], channels=[channel], name=name)

    @property
    def shape(self):
        if self._shape is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(self.file) as src:
                    # todo: support for multiple planes per file?
                    self._shape = (1, src.meta['height'], src.meta['width'], src.meta['count'])
        return self._shape

    def get_data(self, channel=None):
        """Return array of image data.

        If the image has multiple channels, then the name of the channel to return must be given.
        """
        index = self._get_channel_index(channel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return src.read(index)

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion

        Parameters
        ----------
        frames : tuple
            (first_frame, last_frame)
        rows : tuple
            (first_row, last_row+1)
        cols : tuple
            (first_col, last_col+1)
        channel : str | None
            Name of channel to return data from
        """
        index = self._get_channel_index(channel)
        win = rasterio.windows.Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return src.read(index, window=win)

    def _get_channel_index(self, channel):
        # rasterio indexes channels starting at 1
        if channel is None and len(self.channels) == 1:
            return 1
        else:
            return self.channels.index(channel) + 1


class ImageTransform:
    """Transfomation mapping between (x, y) spot coordinates and (row, col) image pixels
    
    *matrix* is a numpy array of shape (2, 3) containing the affine transformation matrix
    """
    def __init__(self, matrix):
        self.matrix = matrix
        assert matrix.shape == (2, 3)
        self._inverse = None

    def map_to_pixels(self, points):
        """Map (x, y) positions to image pixels (row, col). 
        
        Points must be an array of shape (N, 2).
        """
        points = np.asarray(points)
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
        pixels = np.asarray(pixels)
        return (self.inverse_matrix[:2, :2] @ pixels.T + self.inverse_matrix[:2, 2:]).T

    def translated(self, offset):
        """Return a new transform that is translated by *offset* (where offset is expressed in pixels)
        """
        m = self.matrix.copy()
        m[:, 2] += offset
        return ImageTransform(m)


class ImageStack(ImageBase):
    """A stack of Image z-planes

    Assumes images are all the same shape and evenly spaced along the z axis.
    """
    def __init__(self, images, name=None):
        super().__init__()
        self.images = images
        self.name = name
        # z0 = self.images[0].transform.map_from_pixels([[

    @classmethod
    def load_merscope_stacks(cls, path):
        """Read standard merscope image mosaic format, returning multiple image stacks
        """
        transform_file = os.path.join(path, 'micron_to_mosaic_pixel_transform.csv')

        # look for TIF files with the structure like "mosaic_DAPI_z3.tif"
        image_files = glob.glob(os.path.join(path, 'mosaic_*_z*.tif'))
        image_meta = {}
        stains = set()
        z_inds = set()
        for filename in image_files:
            m = re.match(r'mosaic_(\S+)_z(\d+).tif', os.path.split(filename)[1])
            if m is None:
                continue
            stain, z_index = m.groups()
            z_index = int(z_index)
            image_meta[(stain, z_index)] = filename
            stains.add(stain)
            z_inds.add(z_index)

        # for each available stain, make a new stack
        z_inds = sorted(list(z_inds))
        stacks = []
        for stain in stains:
            images = []
            for z_ind in z_inds:
                img_file = image_meta[stain, z_ind]
                img = ImageFile.load_merscope(img_file, transform_file, channel=stain)
                images.append(img)
            stacks.append(ImageStack(images))

        return stacks

    @property
    def shape(self):
        img_shape = self.images[0].shape
        return (len(self.images),) + img_shape[1:] 

    @property
    def channels(self):
        return self.images[0].channels

    @property
    def transform(self):
        return self.images[0].transform

    def get_data(self, channel=None):
        def get_image_data():
            for img in self.images:
                yield img.get_data(channel=channel)
        return self._load_data(get_image_data(), self.shape[0])

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        images = self.images[frames[0]:frames[1]]
        def get_image_data():
            for img in images:
                yield img.get_sub_data(frames=None, rows=rows, cols=cols, channel=channel)
        return self._load_data(get_image_data(), len(images))

    def _load_data(self, gen, n_frames):
        # load data to a 3D array one frame at a time
        # (avoiding np.stack() to reduce memory usage)
        first = next(gen)
        full = np.empty((n_frames,) + first.shape, dtype=first.dtype)
        full[0] = first
        del first
        for i, img in enumerate(gen):
            full[i+1] = img
        return full


class ImageView(ImageBase):
    """Represents a subset of data from an Image (a rectangular subregion or subset of channels)
    """
    def __init__(self, image, frames=None, rows=None, cols=None, channels=None):
        super().__init__()
        if channels is not None:
            for ch in channels:
                assert ch in image.channels
        self.image = image
        self.view_frames = frames or (0, image.shape[0])
        self.view_rows = rows or (0, image.shape[1])
        self.view_cols = cols or (0, image.shape[2])

        for ax, rgn in enumerate([self.view_frames, self.view_rows, self.view_cols]):
            assert rgn[0] >= 0
            assert rgn[1] >= rgn[0]
            assert rgn[1] <= image.shape[ax]

        self.view_channels = channels

        offset = np.zeros(2)
        if rows is not None:
            offset[0] = rows[0]
        if cols is not None:
            offset[1] = cols[0]

        self.transform = image.transform.translated(-offset)

    @property
    def name(self):
        return self.image.name

    @property
    def shape(self):
        shape = [
            self.view_frames[1] - self.view_frames[0],
            self.view_rows[1] - self.view_rows[0],
            self.view_cols[1] - self.view_cols[0],
            self.image.shape[3]
        ]
        if self.view_channels is not None:
            shape[3] = len(self.view_channels)
        return tuple(shape)

    @property
    def channels(self):
        if self.view_channels is not None:
            return self.view_channels
        else:
            return self.image.channels

    def get_data(self, channel=None):        
        return self.image.get_sub_data(self.view_frames, self.view_rows, self.view_cols, channel=channel)

    def get_sub_data(self, frames, rows, cols, channel=None):
        framestart = self.view_frames[0]
        rowstart = self.view_rows[0]
        colstart = self.view_cols[0]
        frames = (frames[0] + framestart, frames[1] + framestart)
        rows = (rows[0] + rowstart, rows[1] + rowstart)
        cols = (cols[0] + colstart, cols[1] + colstart)
        return self.image.get_sub_data(frames, rows, cols, channel=channel)

