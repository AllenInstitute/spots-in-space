from __future__ import annotations
import warnings
import os, glob, re
import numpy as np
from .optional_import import optional_import
rasterio = optional_import('rasterio')
tifffile = optional_import('tifffile')


class ImageBase:
    """Base class representing a 4D image (frames, rows, columns, channels)
    
    Defines basic methods for navigating and displaying the underlying data, but does not define how it should be read in or stored.
    """
    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        """
        raise NotImplementedError()

    def get_data(self, channel=None):
        raise NotImplementedError()

    def get_channel(self, channel: str):
        """Return a view of this image limited to the given channel (e.g.: 'DAPI' or 'PolyT')

        Parameters
        ----------
        channel : str
            Name of channel to return data from
        
        Returns
        ------- 
        ImageView
            A view of this image limited to the channel
        """
        assert channel in self.channels
        if len(self.channels) == 1:
            return self
        return ImageView(self, channels=[channel])
        
    def get_subregion(self, region: tuple|list[tuple]|list[list], incl_end=False):
        """Return a view of this image limited to the spot coordinate-defined region ((xmin, xmax), (ymin, ymax))
        
        Parameters
        ----------
        region : tuple or list[tuple] or list[list]
            Region to return data from
        incl_end : bool
            Include all pixels of the image that overlap with the region, rather than just those inside the region.
            
        Returns
        -------
        ImageView
            A view of this image limited to the region
        """
        corners = np.array(region).T
        tl, br = self.transform.map_to_pixels(corners) # tl is top-left (xmin, ymin), br is bottom-right (xmax, ymax)
        tl = tl.astype(int)
        br = np.ceil(br).astype(int) if incl_end else br.astype(int)
        return ImageView(self, rows=(tl[0],br[0]), cols=(tl[1],br[1]))

    def bounds(self):
        """Return (xlim, ylim) bounds of the image in spot coordinates
        
        Returns
        -------
        tuple
        """
        o = self.transform.map_from_pixels([[0, 0]])[0]
        c = self.transform.map_from_pixels([self.shape[1:3]])[0]
        return ((o[0], c[0]), (o[1], c[1]))

    def get_frame(self, frame: int):
        """Return a view of this image limited to the given frame
        
        Parameters
        ----------
        frame : int
            Frame to return data from
        
        Returns
        -------
        ImageView
        """
        return self.get_frames((frame, frame+1))

    def get_frames(self, frames: tuple):
        """
        Return a view of this image limited to the given frames
        
        Parameters
        ----------
        frames : tuple
            first_frame is inclusive, last_frame is exclusive]
        
        Returns
        -------
        ImageView
        """
        z_len = self.shape[0]
        assert frames[1] <= z_len
        return ImageView(self, frames=frames)

    def show(self, ax=None, frame: int|str|None=None, channel: str=None, **kwds):
        """Show the image in a matplotlib axis
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis to show image in
        frame : int or str or None, optional
            Single frame to show. Can be 'mean' to show the mean of all frames
        channel : str or None, optional
            Channel to show
        **kwds : dict[str, Any]
            Additional keyword arguments to pass to imshow()
        """
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
            data = data[0] # can only specify 1 frame, so guaranteed idx=0

        self._show_image(data, ax, **kwds)

    def _show_image(self, data, ax, **kwds):
        """Takes data directly and shows it on given maptlotlib axis.
        
        This allows a level of abstraction from show, which defines how to get the data/axis
        
        Parameters
        ----------
        data : np.ndarray
            Data to show
        ax : matplotlib.axes.Axes
            Axis to show image in
        **kwds : dict[str, Any]
            Additional keyword arguments to pass to imshow()
        """
        if len(data.shape)==3:
            if data.shape[0] ==1:
                data = data[0,:,:]
            elif data.shape[0] != 3:
                raise ValueError("show requires 2d image data or RGB data with shape [3,N,M]")
        y_inverted = ax.yaxis_inverted()
        shape = self.shape
        px_corners = np.array([[0, 0], shape[1:3]])
        (left, top), (right, bottom) = self.transform.map_from_pixels(px_corners)
        kwds['extent'] = (left, right, bottom, top) # properly labels xy coordinates in shown plot rather than pixel coordinates
        ax.imshow(data, **kwds)
        # ax.yaxis_inverted() will return True after imshow, so we compare it against original axis
        # Original axis generally will have not be inverted, this will create a != situation 
        # and we will invert axes to get back to ax.yaxis_inverted()==False
        # If user already had original axis inverted, then its okay, we can keep it inverted
        if ax.yaxis_inverted() != y_inverted:
            ax.invert_yaxis()


class Image(ImageBase):
    """An Image defined by 4D numpy array (frames, rows, cols, channels).
    
    Carries metadata about pixel transform and channel identity.
    
    Attributes
    ----------
    transform : ImageTransform
        Transformation mapping from pixel coordinates to spot coordinates
    channels : list
        List of names given to each channel (e.g.: 'dapi')
    name : str or None
        Optional unique identifier for this image
    _data : np.ndarray
        4D array of image data
    """
    def __init__(self, data:np.ndarray, transform:ImageTransform, channels:list, name: str|None=None):
        super().__init__()
        """
        Parameters
        ----------
        data : np.ndarray
            4D array of image data
        transform : ImageTransform
            Transformation mapping from pixel coordinates to spot coordinates
        channels : list
            List of names given to each channel (e.g.: 'dapi')
        name : str or None, optional
            Optional unique identifier for this image
        """
        assert data.ndim == 4
        assert isinstance(transform, ImageTransform)
        self.transform = transform
        self.channels = channels
        self.name = name
        self._data = data

    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        
        Returns
        -------
        tuple
        """
        return self._data.shape
        
    def get_data(self, channel=None):
        """Return array of image data.
        
        If the image has multiple channels, then the name of the channel to return must be given.
        
        Parameters
        ----------
        channel : str or None, optional
            Name of channel to return data from
        """
        index = self._get_channel_index(channel)
        return self._data[..., index]

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : tuple
            first frame is inclusive, last_frame is exclusive
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
        """
        chan = self.get_data(channel)
        return chan[frames[0]:frames[1], rows[0]:rows[1], cols[0]:cols[1]]

    def _get_channel_index(self, channel):
        """Return the index of the given channel in the underlying data array
        
        Parameters
        ----------
        channel : str or None
            Name of channel to return data from
        """
        if channel is not None:
            return self.channels.index(channel)
        else:
            # If there is more than one channel force user to specify, otherwise just return the one channel
            assert self.shape[3] == 1, "Must specify channel to return"
            return 0


class ImageFile(ImageBase):
    """Represents a single image stored on disk, carrying metadata about:
    - The file containing image data
    - The transform that maps from pixel coordinates to spot table coordinates
    - Which axes are which
    - What is represented by each channel

    Image data are lazy-loaded so that we can handle subregions without loading the entire image
    
    Attributes
    ----------
    file : str
        Path to image file
    transform : ndarray
        ImageTransform relating (row, col) image pixel coordinates to (x, y) spot coordinates.
    axes : list
        List of axis names giving order of axes in *file*; options are 'frame', 'row', 'col', 'channel'
    channels : list
        List of names given to each channel (e.g.: 'dapi')
    name : str or None
        Optional unique identifier for this image
    _shape : tuple or None
        Cached shape of the image, to avoid reading it from disk multiple times
    """
    def __init__(self, file: str, transform:ImageTransform, axes: list|None, channels: list, name: str|None):
        """
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
        name : str or None
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
        """Read a Merscope image file and return an ImageFile object
        
        Parameters
        ----------
        image_file : str
            Path to image file
        transform_file : str
            Path to transform file
        channel : str
            Name of channel which this image represents
        name : str or None, optional
            Optional unique identifier for this image
        """
        um_to_px = np.loadtxt(transform_file)[:2]
        # swizzle first and second rows so we map from (x, y) to (row,col) instead of (col,row)
        # since images take (row, col) as coordinates
        um_to_px = um_to_px[::-1]
        tr = ImageTransform(um_to_px)
        return ImageFile(file=image_file, transform=tr, axes=['frame', 'row', 'col', 'channel'], channels=[channel], name=name)

    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        
        Returns
        -------
        tuple
        """
        if self._shape is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(self.file) as src:
                    # todo: support for multiple planes per file?
                    self._shape = (1, src.meta['height'], src.meta['width'], src.meta['count'])
        return self._shape

    def _standard_image_shape(self, img_data):
        """Convert image data to standard 1-channel shape (frames, rows, columns)
        
        Returns
        -------
        np.ndarray
        """
        if img_data.ndim == 2:
            img_data = img_data[np.newaxis, :,:]
        return img_data

    def get_data(self, channel=None):
        """Return array of image data.
        
        If the image has multiple channels, then the name of the channel to return must be given.
        
        Parameters
        ----------
        channel : str or None, optional
            Name of channel to return data from
        """
        index = self._get_channel_index(channel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return self._standard_image_shape(src.read(index))

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : tuple
            first frame is inclusive, last_frame is exclusive
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
        """
        index = self._get_channel_index(channel)
        win = rasterio.windows.Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return self._standard_image_shape(src.read(index, window=win))

    def _get_channel_index(self, channel):
        """Return the index of the given channel in the underlying data array
        
        Parameters
        ----------
        channel : str or None
            Name of channel to return data from
        """
        # rasterio indexes channels starting at 1
        if channel is None and len(self.channels) == 1:
            return 1
        else:
            return self.channels.index(channel) + 1


class ImageTransform:
    """Transfomation mapping between (x, y) spot coordinates and (row, col) image pixels
    
    Parameters
    ----------
    matrix : np.ndarray
        2x3 affine transformation matrix mapping from spots to pixels
    """
    def __init__(self, matrix):
        self.matrix = matrix
        assert matrix.shape == (2, 3)
        self._inverse = None

    def map_to_pixels(self, points):
        """Map (x, y) positions to image pixels (row, col). 
        
        Parameters
        ----------
        points : np.ndarray
            Points to map. Must be of shape (N, 2)
        """
        points = np.asarray(points)
        assert points.ndim == 2
        assert points.shape[1] == 2
        return (self.matrix[:2, :2] @ points.T + self.matrix[:2, 2:]).T

    @property
    def inverse_matrix(self):
        """Returns the inverse of the transformation matrix
        
        Returns
        -------
        np.ndarray
        """
        if self._inverse is None: # if we haven't stored inverse, calculate it
            m3 = np.eye(3)
            m3[:2] = self.matrix
            self._inverse = np.linalg.inv(m3)[:2]
        return self._inverse

    def map_from_pixels(self, pixels):
        """Map (row, col) image pixels to positions (x, y). 
        
        Parameters
        ----------
        pixels : np.ndarray
            Pixels to map. Must be of shape (N, 2)
            
        Returns
        -------
        np.ndarray
        """
        pixels = np.asarray(pixels)
        assert pixels.ndim == 2
        assert pixels.shape[1] == 2
        return (self.inverse_matrix[:2, :2] @ pixels.T + self.inverse_matrix[:2, 2:]).T

    def translated(self, offset):
        """Return a new transform that is translated by *offset* (where offset is expressed in pixels)
        
        Parameters
        ----------
        offset : np.ndarray
            Offset to translate by. Must be of shape (2,)
        """
        m = self.matrix.copy()
        m[:, 2] += offset
        return ImageTransform(m)


class XeniumImageFile(ImageBase):
    """Represents a single image stored on disk, carrying metadata about:
    - The file containing image data
    - The transform that maps from pixel coordinates to spot table coordinates
    - Which axes are which
    - What is represented by each channel
    Image data are lazy-loaded so that we can handle subregions without loading the entire image
    
    Attributes
    ----------
    file : str
        Path to image file
    transform : ndarray
        ImageTransform relating (row, col) image pixel coordinates to (x, y) spot coordinates.
    axes : list or None
        List of axis names giving order of axes in *file*; options are 'frame', 'row', 'col', 'channel'
    channels : list
        List of names given to each channel (e.g.: 'dapi')
    z_index : int
        Xenium images come with all z-planes in one image. This is the index of the z-plane to load.
    name : str or None
        Optional unique identifier for this image
    pyramid_level : int
        Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
        This is not currently utilized anywhere, but included for potential future use.
    _shape : tuple or None
        Cached shape of the image, to avoid reading it from disk multiple times
    whole_image_array : np.ndarray or None
        Cached array of the whole image data, to avoid reading it from disk multiple times
    keep_images_in_memory : bool
        Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. The trade off is speed vs memory.
    """
    
    def __init__(self, file: str, transform: ImageTransform, axes: list|None,
                  channels: list, z_index:int, name: str|None,
                  pyramid_level: int= 0, keep_images_in_memory: bool = True):
        """
        Parameters
        ----------
        file : str
            Path to image file
        transform : ndarray
            ImageTransform relating (row, col) image pixel coordinates to (x, y) spot coordinates.
        axes : list or None
            List of axis names giving order of axes in *file*; options are 'frame', 'row', 'col', 'channel'
        channels : list
            List of names given to each channel (e.g.: 'dapi')
        z_index : int
            Xenium images come with all z-planes in one image. This is the index of the z-plane to load.
        name : str or None
            Optional unique identifier for this image
        pyramid_level : int, optional
            Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
            This is not currently utilized anywhere, but included for potential future use.
        keep_images_in_memory : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. The trade off is speed vs memory.
        """
        super().__init__()
        self.file = file
        self.transform = transform
        self.axes = axes
        self.channels = channels
        self.z_index = z_index  # hold this constant within each "XeniumImageFile" for now
        self.name = name
        self.pyramid_level = pyramid_level
        self._shape = None
        self.whole_image_array = None
        self.keep_images_in_memory = keep_images_in_memory

    @classmethod
    def load_xenium(cls, image_file, transform_matrix, z_index, channel = "DAPI", name = None, pyramid_level = 0, keep_images_in_memory = True):
        """Read a Xenium image file and return an XeniumImageFile object. 
        
        Parameters
        ----------
        image_file : str
            Path to image file
        transform_matrix : np.ndarray
            2x3 affine transformation matrix mapping from spots to pixels
        z_index : int
            Xenium images come with all z-planes in one image. This is the index of the z-plane to load.
        channel : str
            Name of channel which this image represents
        name : str or None, optional
            Optional unique identifier for this image
        pyramid_level : int, optional
            Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
            This is not currently utilized anywhere, but included for potential future use.
        keep_images_in_memory : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. 
            The trade off is speed vs memory.
            
        Returns
        -------
        XeniumImageFile
            An XeniumImageFile object
        """
        # swizzle first and second rows so we map from (x, y) to (row,col) instead of (col,row)
        # since images take (row, col) as coordinates
        tr = ImageTransform(transform_matrix[::-1])
        return XeniumImageFile(file=image_file, transform=tr, axes=['frame', 'row', 'col', 'channel'],
                                channels=[channel], z_index= z_index, name=name,
                                pyramid_level = pyramid_level, keep_images_in_memory=keep_images_in_memory)

    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        
        Returns
        -------
        tuple
        """
        if self._shape is None: # Caching helps speed up retrieval of shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with tifffile.TiffFile(self.file) as tcontext:
                    self._shape = (1, tcontext.series[0].shape[1], tcontext.series[0].shape[2], 1)
        return self._shape

    def _standard_image_shape(self, img_data):
        """Convert image data to standard 1-channel shape (frames, rows, columns)
        """
        if img_data.ndim == 2:
            img_data = img_data[np.newaxis, :,:]
        return img_data

    def get_data(self, channel=None):
        """Return array of image data.
        Channel name is left for support, but not currently used.
        """
        #index = self._get_channel_index(channel)
        if isinstance( self.whole_image_array, type(None)): # if it's not cached, we have to read
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with tifffile.TiffFile(self.file) as src:
                    if self.keep_images_in_memory:
                        # since the Xenium images are large, we leave it as an option whether to cache the images or not
                        self.whole_image_array = self._standard_image_shape(src.series[0][self.z_index].asarray())
                    else:
                        return self._standard_image_shape(src.series[0][self.z_index].asarray())
        return self.whole_image_array

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : tuple
            first frame is inclusive, last_frame is exclusive
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
        
        Returns
        -------
        np.ndarray
            Subregion of image data
        """
        # index = self._get_channel_index(channel)

        if self.keep_images_in_memory:
            if isinstance(self.whole_image_array, type(None)):
                self.get_data(channel = channel)
            return self._standard_image_shape(self.whole_image_array[0][rows[0]:rows[1],cols[0]:cols[1]])
        else:
            return self._standard_image_shape(self.get_data(channel = channel)[0][rows[0]:rows[1],cols[0]:cols[1]])

    def _get_channel_index(self, channel):
        """Return the index of the given channel in the underlying data array
        
        Parameters
        ----------
        channel : str or None
            Name of channel to return data from
        """
        if channel is None and len(self.channels) == 1:
            return 0
        else:
            return self.channels.index(channel) 


class ImageStack(ImageBase):
    """A stack of Image z-planes
    Assumes images are all the same shape and evenly spaced along the z axis.
    
    Attributes
    ----------
    images : list
        List of Image objects
    name : str or None
        Optional unique identifier for this image
    """
    def __init__(self, images, name=None):
        """
        Parameters
        ----------
        images : list
            List of Image objects
        name : str or None, optional
            Optional unique identifier for this image
        """
        super().__init__()
        self.images = images
        self.name = name

    @classmethod
    def load_merscope_stacks(cls, path):
        """Read standard merscope image mosaic format, returning multiple image stacks
        We read merscope into ImageStacks because z-planes are stored across different files.
        
        Parameters
        ----------
        path : str
            Path to directory containing image files
        
        Returns
        -------
        stacks : list
            List of ImageStacks, one for each stain
        """
        transform_file = os.path.join(path, 'micron_to_mosaic_pixel_transform.csv')

        # look for TIF files with the structure like "mosaic_DAPI_z3.tif" or "mosaic_PolyT_z3.tif"
        image_files = glob.glob(os.path.join(path, 'mosaic_*_z*.tif'))
        image_meta = {}
        stains = set()
        z_inds = set()
        for filename in image_files:
            # Pull stain and z-index from filename
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

    @classmethod
    def load_xenium_stacks(cls, xenium_image_file, pyramid_to_keep=None, max_z_to_take=None, keep_images_in_memory=True):
        """Read standard Xenium image mosaic tiff file, returning list of XeniumImageFiles
        
        Parameters
        ----------
        xenium_image_file : str
            Path to image file
        pyramid_to_keep : int or None, optional
            Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
            This is not currently utilized anywhere, but included for potential future use.
        max_z_to_take : int or None, optional
            Xenium images come with all z-planes in one image. We will only load z-planes up to this index
            Can help with get_data() memory usage by ignoring z-planes that are not used.
        keep_images_in_memory : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. 
            The trade off is speed vs memory.
            
        Returns
        -------
        stacks : list
            List of ImageStacks, one for each stain
        """
        import xml.etree.ElementTree as ET

        tiff_image_file = tifffile.TiffFile(xenium_image_file)

        # this file should have OME metadata:
        metadata_root = ET.fromstring(tiff_image_file.ome_metadata)
        # extract the pixel size 
        for child in metadata_root:
            if "Image" in child.tag:
                for cc in child:
                    if "Pixels" in cc.tag:
                        um_per_pixel_x = float(cc.attrib["PhysicalSizeX"])
                        um_per_pixel_y = float(cc.attrib["PhysicalSizeY"])
        # and turn this into a transformation matrix
        affine_matrix = np.eye(3)[:2,:]
        affine_matrix[0,0] = um_per_pixel_x
        affine_matrix[1,1] = um_per_pixel_y
        m3 = np.eye(3)
        m3[:2] = affine_matrix
        um_to_pixel_matrix =  np.linalg.inv(m3)[:2]

        # the Xenium OME-TIFF file is an image pyramid. 
        #I'm basing everything here on the highest resolution level.
        image_file_shape = tiff_image_file.series[0].shape

        z_inds = list(range(image_file_shape[0]))
        if max_z_to_take:
            z_inds = z_inds[:min(max_z_to_take, image_file_shape[0])]
        # leave this list of `stacks` to account for future versions with multiple stains
        stacks = []
        for stain in ["DAPI"]:
            images = []
            for z_ind in z_inds:
                img = XeniumImageFile.load_xenium(xenium_image_file, um_to_pixel_matrix, z_index=z_ind, channel=stain, pyramid_level=pyramid_to_keep, keep_images_in_memory=keep_images_in_memory)
                images.append(img)
            stacks.append(ImageStack(images))

        return stacks


    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        
        Returns
        -------
        tuple
        """
        img_shape = self.images[0].shape # All images are assumed to be the same shape so we just query the first
        return (len(self.images),) + img_shape[1:] 

    @property
    def channels(self):
        """Return list of channel names
        
        Returns
        -------
        list[str]
        """
        return self.images[0].channels # All images are assumed to be the same shape so we just query the first

    @property
    def transform(self):
        """Return transform mapping from pixel coordinates to spot coordinates
        
        Returns
        -------
        sis.image.ImageTransform
        """
        return self.images[0].transform # All images are assumed to be the same shape so we just query the first

    def get_data(self, channel=None):
        """Return array of image data.
        
        Parameters
        ----------
        channel : str or None, optional
            Name of channel to return data from
        
        Returns
        -------
        np.ndarray
            3D array of image data (frames, rows, columns)
        """
        def get_image_data():
            for img in self.images:
                yield img.get_data(channel=channel)
        return self._load_data(get_image_data(), self.shape[0])

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)
        
        Parameters
        ----------
        frames : tuple
            first frame is inclusive, last_frame is exclusive
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from      
            
        Returns
        -------
        np.ndarray
            3D array of image data (frames, rows, columns)
        """
        images = self.images[frames[0]:frames[1]]
        def get_image_data():
            for img in images:
                yield img.get_sub_data(frames=None, rows=rows, cols=cols, channel=channel)
        return self._load_data(get_image_data(), len(images))

    def _load_data(self, gen, n_frames):
        """Helper function to load data from a generator into a 3D array
        We load data to a 3D array one frame at a time. Avoiding np.stack() to reduce memory usage
        
        Parameters
        ----------
        gen : generator
            Generator yielding image data
        n_frames : int
            Number of frames to load
        """
        first = next(gen)[0]
        full = np.empty((n_frames,) + first.shape, dtype=first.dtype)
        full[0] = first
        del first
        for i, img in enumerate(gen):
            full[i+1] = img[0]
        return full


class ImageView(ImageBase):
    """Represents a subset of data from an Image (a rectangular subregion or subset of channels)
    Tracking the view of the image along with the image allows us to serve up subregions accurately when loading images from disk
    (since we can't just store the subimaged data array in memory) 
    
    Attributes
    ----------
    image : ImageBase
        Image to get data from
    view_frames : tuple
        Frames to include in the view
    view_rows : tuple
        Rows to include in the view
    view_cols : tuple
        Columns to include in the view
    view_channels : list or None
        List of channels to include in the view. If None, all channels are included.
    transform : ImageTransform
        Transformation mapping from pixel coordinates to spot coordinates, adjusted for the view
    """
    def __init__(self, image, frames=None, rows=None, cols=None, channels=None):
        """
        Parameters
        ----------
        image : ImageBase
            Image to get data from
        frames : tuple or None, optional
            Frames to include in the view. If None, all frames are included. (first_frame is inclusive, last_frame is exclusive)
        rows : tuple or None, optional
            Rows to include in the view. If None, all rows are included. (first_row is inclusive, last_row is exclusive)
        cols : tuple or None, optional
            Columns to include in the view. If None, all columns are included. (first_col is inclusive, last_col is exclusive)
        channels : list or None, optional
            List of channels to include in the view. If None, all channels are included.
        """
        super().__init__()
        if channels is not None:
            for ch in channels:
                assert ch in image.channels
        self.image = image
        self.view_frames = frames or (0, image.shape[0])
        self.view_rows = rows or (0, image.shape[1])
        self.view_cols = cols or (0, image.shape[2])

        # Check that we are in bounds of image and that the last index is greater than the first
        for ax, rgn in enumerate([self.view_frames, self.view_rows, self.view_cols]):
            assert rgn[0] >= 0
            assert rgn[1] >= rgn[0]
            assert rgn[1] <= image.shape[ax]

        self.view_channels = channels

        # Update the image transform to account for the view
        offset = np.zeros(2)
        if rows is not None:
            offset[0] = rows[0]
        if cols is not None:
            offset[1] = cols[0]

        self.transform = image.transform.translated(-offset)

    @property
    def name(self):
        """Return optional unique identifier for image
        
        Returns
        -------
        str
            Name of the image, or None if not specified
        """
        return self.image.name

    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels) of the view as a tuple
        
        Returns
        -------
        tuple
        """
        shape = [
            self.view_frames[1] - self.view_frames[0],
            self.view_rows[1] - self.view_rows[0],
            self.view_cols[1] - self.view_cols[0],
            self.image.shape[3] 
        ]
        # channels is equal to the number of channels in the underlying image unless specified
        if self.view_channels is not None:
            shape[3] = len(self.view_channels)
        return tuple(shape)

    @property
    def channels(self):
        """channels is equal to the number of channels in the underlying image unless specified
        
        Returns
        -------
        list[str]
            List of channel names
        """
        if self.view_channels is not None:
            return self.view_channels
        else:
            return self.image.channels

    def get_data(self, channel=None):
        """Return array of image data.
        
        Parameters
        ----------
        channel : str or None, optional
            Name of channel to return data from
                
        Returns
        -------
        np.ndarray
            Subregion of image data
        """
        return self.image.get_sub_data(self.view_frames, self.view_rows, self.view_cols, channel=channel)

    def get_sub_data(self, frames, rows, cols, channel=None):
        """Return a view of this image limited to the given frames, rows, and cols
        
        Parameters
        ----------
        frames : tuple
            first_frame is inclusive, last_frame is exclusive
        rows : tuple
            first_row is inclusive, last_row is exclusive
        cols : tuple
            first_col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
        
        Returns
        -------
        np.ndarray
            Subregion of image data
        """
        framestart = self.view_frames[0]
        rowstart = self.view_rows[0]
        colstart = self.view_cols[0]
        frames = (frames[0] + framestart, frames[1] + framestart)
        rows = (rows[0] + rowstart, rows[1] + rowstart)
        cols = (cols[0] + colstart, cols[1] + colstart)
        return self.image.get_sub_data(frames, rows, cols, channel=channel)

