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

    def get_frame(self, frame: int|None):
        """Return a view of this image limited to the given frame
        
        Parameters
        ----------
        frame : int or None
            Frame to return data from
        
        Returns
        -------
        ImageView
        """
        if frame is None:
            return self
        return self.get_frames((frame, frame+1))

    def get_frames(self, frames: tuple|int|None):
        """
        Return a view of this image limited to the given frames
        
        Parameters
        ----------
        frames : tuple or int or None
            if tuple: first_frame is inclusive, last_frame is exclusive
            if int: single frame to return
            if None: return full image
        
        Returns
        -------
        ImageView
        """
        if frames is None:
            return self
        elif isinstance(frames, int):
            assert frames <= self.shape[0]
            return ImageView(self, frames=(frames, frames+1))
        elif isinstance(frames, tuple):
            assert frames[1] <= self.shape[0]
            return ImageView(self, frames=frames)
        else:
            raise ValueError('Frames must be None, an int, or a tuple of (first_frame, last_frame)')

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
            img = img.get_frames(frame)
            
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
    """Base class for accessing image files on disk.
    """
    def _standard_image_shape(self, img_data):
        """Convert image data to standard 1-channel shape (frames, rows, columns)
        
        Parameters
        ----------
        img_data : np.ndarray
            Image data to convert into the standard shape of (frames, rows, columns)
            
        Returns
        -------
        np.ndarray
            Image data in standard shape
        """
        if img_data.ndim == 2:
            img_data = img_data[np.newaxis, :,:]
        return img_data
    
    @property
    def shape(self):
        """Return 4D shape (frames, rows, columns, channels)
        
        Returns
        -------
        np.ndarray
        """
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
        
        Parameters
        ----------
        channel : str|None, optional
            Name of channel to return data from
    
        Returns
        -------
        np.ndarray
        """
        index = self._get_channel_index(channel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return self._standard_image_shape(src.read(index))

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Return a subset of the ImageData
        
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
        numpy.ndarray
            Subregion of image data
        """
        return self.get_data(channel=channel)[frames[0]:frames[1], rows[0]:rows[1], cols[0]:cols[1]]

    def _get_channel_index(self, channel: str|None=None):
        """
        Return the index of the given channel in the underlying data array
        
        Parameters
        ----------
        channel : str or None
            Name of channel to return data from
            
        Returns
        -------
        int
        """
        # rasterio indexes channels starting at 1
        if channel is None and len(self.channels) == 1:
            return 1
        else:
            return self.channels.index(channel) + 1
    

class MerscopeImageFile(ImageFile):
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
    def load(cls, image_file, transform_file, channel, name=None):
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
            
        Returns
        -------
        MerscopeImageFile
        """
        um_to_px = np.loadtxt(transform_file)[:2]
        # swizzle first and second rows so we map from (x, y) to (row,col) instead of (col,row)
        # since images take (row, col) as coordinates
        um_to_px = um_to_px[::-1]
        tr = ImageTransform(um_to_px)
        return MerscopeImageFile(file=image_file, transform=tr, axes=['frame', 'row', 'col', 'channel'], channels=[channel], name=name)

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : None
            not used for merscope image files since they are split by frame already, but left for compatibility
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
            
        Returns
        -------
        numpy.ndarray
        """
        index = self._get_channel_index(channel)
        win = rasterio.windows.Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return self._standard_image_shape(src.read(index, window=win))

class StereoSeqImageFile(ImageFile):
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
    def load(cls, image_file, xyscale, channel, name=None):
        """Read a StereoSeq image file and return an ImageFile object
        
        Parameters
        ----------
        image_file : str
            Path to image file
        xyscale : float
            um/pixel scale factor used to create the transform matrix
        channel : str
            Name of channel which this image represents
        name : str or None, optional
            Optional unique identifier for this image
            
        Returns
        -------
        StereoSeqImageFile
        """
        # values are off diagonal so we map from (x, y) to (row,col) instead of (col,row)
        # since images take (row, col) as coordinates
        transform = ImageTransform(matrix=np.array([
            [0, 1/xyscale, 0],
            [1/xyscale, 0, 0],
        ], dtype=float))
        return StereoSeqImageFile(file=image_file, transform=transform, axes=['frame', 'row', 'col', 'channel'], channels=[channel], name=name)

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, channel: str|None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : None
            not used for stereoseq image files since they are 2d
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        channel : str or None, optional
            Name of channel to return data from
            
        Returns
        -------
        numpy.ndarray
        """
        index = self._get_channel_index(channel)
        win = rasterio.windows.Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with rasterio.open(self.file) as src:
                return self._standard_image_shape(src.read(index, window=win))


class ImageTransform:
    """Transfomation mapping between (x, y) spot coordinates and (row, col) image pixels
    
    Attributes
    ----------
    matrix : np.ndarray
        2x3 affine transformation matrix mapping from spots to pixels
    _inverse : np.ndarray or None
        Cached inverse of the transformation matrix
    """
    def __init__(self, matrix):
        """
        Parameters
        ----------
        matrix : np.ndarray
            2x3 affine transformation matrix mapping from spots to pixels
        """
        self.matrix = matrix
        assert matrix.shape == (2, 3)
        self._inverse = None

    def map_to_pixels(self, points):
        """Map (x, y) positions to image pixels (row, col). 
        
        Parameters
        ----------
        points : np.ndarray
            Points to map. Must be of shape (N, 2)
            
        Returns
        -------
        numpy.ndarray
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
            
        Returns
        -------
        ImageTransform
            New transform that is translated by *offset*
        """
        m = self.matrix.copy()
        m[:, 2] += offset
        return ImageTransform(m)

    @classmethod
    def load_spatialdata_transformation(cls, transformation: sd.transformations.BaseTransformation):
        """Load a spatial data transformation and convert it to an ImageTransform.

        Parameters
        ----------
        cls : type
            ImageTransformation class
        transformation : sd.transformations.BaseTransformation
            The transformation to be converted

        Returns
        -------
        ImageTransform
            conversion of the input spatialdata transformation to an ImageTransform.

        Raises
        ------
        ValueError
            If the input axes of the transformation are not supported (must be (x, y) or (x, y, z)).
        """
        from spatialdata.transformations import Affine
        
        if not isinstance(transformation, Affine):
            transformation = transformation.to_affine(input_axes=('x', 'y'), output_axes=('x', 'y'))
        
        if transformation.input_axes == ('x', 'y'):
            um_to_px = transformation.matrix[:2] # We only use the first 2 rows
        elif transformation.input_axes == ('x', 'y', 'z'):
            um_to_px = transformation.matrix[1:3][:, [0, 1, 3]] # cutting out the z row/column
        else:
            raise ValueError('Unsupported transformation axes. Must be (x,y) or (x,y,z)')
        um_to_px = um_to_px[::-1] # swizzle first and second rows so we map from (x, y) to (row,col) instead of (col,row) since images take (row, col) as coordinates
        return cls(um_to_px)

class XeniumImageFile(ImageFile):
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
    name : str or None
        Optional unique identifier for this image
    pyramid_level : int
        Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
        This is not currently utilized anywhere, but included for potential future use.
    _shape : tuple or None
        Cached shape of the image, to avoid reading it from disk multiple times
    whole_image_array : np.ndarray or None
        Cached array of the whole image data, to avoid reading it from disk multiple times
    cache_image : bool
        Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. The trade off is speed vs memory.
    """
    
    def __init__(self, file: str, transform: ImageTransform, axes: list|None,
                 channels: list, name: str|None,
                 pyramid_level: int= 0, cache_image: bool = True):
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
        name : str or None
            Optional unique identifier for this image
        pyramid_level : int, optional
            Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
            This is not currently utilized anywhere, but included for potential future use.
        cache_image : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. The trade off is speed vs memory.
        """
        super().__init__()
        self.file = file
        self.transform = transform
        self.axes = axes
        self.channels = channels
        self.name = name
        self.pyramid_level = pyramid_level
        self._shape = None
        self.whole_image_array = None
        self.cache_image = cache_image

    @classmethod
    def load(cls, xenium_image_file, pyramid_level=0, cache_image=True, name=None):
        """Read standard Xenium image mosaic tiff file, returning list of XeniumImageFiles
        
        Parameters
        ----------
        xenium_image_file : str
            Path to image file
        pyramid_level : int, optional
            Xenium images are stored as OMEs which support image pyramids. This is the level of the pyramid to load.
            This is not currently utilized anywhere, but included for potential future use.
        cache_image : bool, optional
            Xenium images are large and not memory mapped and thus we may want to keep them in memory or not. 
            The trade off is speed vs memory.
        name : str or None, optional
            Optional unique identifier for this image
            
        Returns
        -------
        stacks : list
            List of ImageStacks, one for each stain
        """
        import xml.etree.ElementTree as ET

        # extract the pixel size 
        with tifffile.TiffFile(xenium_image_file) as tiff_image_file:
            # this file should have OME metadata:
            root = ET.fromstring(tiff_image_file.ome_metadata)
        pixels = cls._find_by_localname(root, "Pixels")
        um_per_pixel_x, um_per_pixel_y = float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])

        # and turn this into a transformation matrix
        affine_matrix = np.eye(3)
        affine_matrix[0,0] = um_per_pixel_x
        affine_matrix[1,1] = um_per_pixel_y
        um_to_pixel_matrix =  np.linalg.inv(affine_matrix)[:2]
        
        # swizzle first and second rows so we map from (x, y) to (row,col) instead of (col,row)
        # since images take (row, col) as coordinates
        transform = ImageTransform(um_to_pixel_matrix[::-1])

        return XeniumImageFile(file=xenium_image_file, transform=transform, 
                               axes=['frame', 'row', 'col', 'channel'], channels=['DAPI'],
                               name=name,  pyramid_level=pyramid_level, cache_image=cache_image)
        
    @classmethod
    def _find_by_localname(cls, root, localname):
        """Helper function to find first XML element with given localname (ignoring namespace)
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Root XML element to search
        localname : str
            Local name of element to find
            
        Returns
        -------
        xml.etree.ElementTree.Element or None
            First element with given localname, or None if not found
        """
        for elem in root.iter():
            if elem.tag.split('}')[-1] == localname:
                return elem
        return None

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
                    # Query the file metadata for the shape
                    import xml.etree.ElementTree as ET
                    pixels = self._find_by_localname(ET.fromstring(tcontext.ome_metadata), 'Pixels')
                    self._shape = (int(pixels.attrib['SizeZ']), int(pixels.attrib['SizeY']), int(pixels.attrib['SizeX']), int(pixels.attrib['SizeC']))
        return self._shape

    def get_data(self, channel=None, pyramid_level=None):
        """Return array of image data.
        
        Parameters
        ----------
        pyramid_level : int or None, optional
            Xenium images can have multiple resolutions stored in an image pyramid.
            This parameter specifies which level of the pyramid to load.
            
        Returns
        -------
        np.ndarray
            Image data at specified pyramid level
        """
        if pyramid_level is None:
            pyramid_level = self.pyramid_level
                        
        if isinstance( self.whole_image_array, type(None)): # if it's not cached, we have to read
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with tifffile.TiffFile(self.file) as src:
                    if self.cache_image:
                        # since the Xenium images are large, we leave it as an option whether to cache the images or not
                        self.whole_image_array = self._standard_image_shape(src.series[0].levels[pyramid_level].asarray())
                    else:
                        return self._standard_image_shape(src.series[0].levels[pyramid_level].asarray())
        return self.whole_image_array

    def get_sub_data(self, frames: tuple, rows: tuple, cols: tuple, pyramid_level: int|None=None, channel: None=None):
        """Get image data for a subregion (defined by frames, rows, and cols, NOT by spot coordinates)

        Parameters
        ----------
        frames : tuple
            first frame is inclusive, last_frame is exclusive
        rows : tuple
            first row is inclusive, last_row is exclusive
        cols : tuple
            first col is inclusive, last_col is exclusive
        pyramid_level : int|None, optional
            Xenium images can have multiple resolutions stored in an image pyramid.
            This parameter specifies which level of the pyramid to load.
        channel : None
            This is not used for xenium data as we only get DAPI
            
        Returns
        -------
        np.ndarray
            Subregion of image data
        """
        if pyramid_level is None:
            pyramid_level = self.pyramid_level
            
        return self._standard_image_shape(self.get_data(pyramid_level=pyramid_level)[frames[0]:frames[1], rows[0]:rows[1], cols[0]:cols[1]])

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
                img = MerscopeImageFile.load(img_file, transform_file, channel=stain)
                images.append(img)
            stacks.append(ImageStack(images))

        return stacks

    @classmethod
    def load_spatialdata_stacks(cls, sd_object, image_names=None, points_name=None):
        """Load images from a spatialdata object into an ImageStack

        Requires that images are named with _z{index} suffixes to indicate z-layer.

        Parameters
        ----------
        cls : type
            The class type that is calling this method.
        sd_object : spatialdata.SpatialData
            The spatial data object containing images
        image_names : list of str, optional
            A list of image names to be loaded. If None, all images in the
            spatial data object will be used.
        points_name : str, optional
            The name of the points element to be used for transformations.
            Defaults to the first points value

        Returns
        -------
        cls
            An ImageStacks instance containing the loaded images.

        Raises
        ------
        ValueError
            If there are duplicate z-layers in the images or if unsupported
            transformations (rotation or shear) are detected.
        """
        import spatialdata as sd
        from spatialdata.transformations import get_transformation
        from .spatialdata import _is_supported_transformation

        
        if image_names is None:
            # By default we stack all images
            image_names = list(sd_object.images.keys())
            
        if points_name is None:
            # Load the first points element by default
            if len(sd_object.points.keys()) > 1:
                import warnings
                warnings.warn('Points name was left unspecified and there are multiple Points elements. Defaulting to the first listed for creating the transformation')
            points_name = list(sd_object.points.keys())[0]
            
        # Extract z-indices from image names so that we can loop over them in order
        z_inds = sorted([int(image_name.split('_z')[-1]) for image_name in image_names])
        if list(set(z_inds)) != z_inds:
            raise ValueError('Duplicate z-layer in images')
        name_dict = {int(image_name.split('_z')[-1]): image_name for image_name in image_names}

        images = []
        for z in z_inds:
            image_name = name_dict[z]
            channels = list(sd.get_pyramid_levels(sd_object[image_name], n=0).c.to_numpy()) # Extract channel names
            
            # Check that the transformation is supported (no rotation or shear)
            if not _is_supported_transformation(get_transformation(sd_object[image_name])):
                raise ValueError('We do not support rotation or shear transformations')
            
            # Load the image and its associated transformation into SIS
            transform = ImageTransform.load_spatialdata_transformation(get_transformation(sd_object[points_name]))
            images.append(SpatialDataImage(sd.get_pyramid_levels(sd_object[image_name], n=0),
                                           transform,
                                           ['frame', 'row', 'col', 'channel'],
                                           channels,
                                           None))
            
        return cls(images)

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
        if channel is None and self.view_channels is not None: # If the user doesn't specify a channel but we have view_channels, then we will just pull from those
            assert len(self.view_channels) == 1, "Must specify channel to return" # across all images, get_data only supports returning one channel
            channel = self.view_channels[0]
        if channel is not None and self.view_channels is not None:
            assert channel in self.view_channels, "Requested channel not in view channels"
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
        if channel is None and self.view_channels is not None: # If the user doesn't specify a channel but we have view_channels, then we will just pull from those
            assert len(self.view_channels) == 1, "Must specify channel to return" # across all images, get_data only supports returning one channel
            channel = self.view_channels[0]
        if channel is not None and self.view_channels is not None:
            assert channel in self.view_channels, "Requested channel not in view channels"
        return self.image.get_sub_data(frames, rows, cols, channel=channel)


class SpatialDataImage(ImageBase):
    """Class for handling spatialdata images within SIS
    The data input is an xarray DataArray
    
    Attributes
    ----------
    transform : callable
        A transformation function to apply to the image data.
    axes : list of str
        The names of the axes corresponding to the dimensions of the data.
    channels : list of str
        The names of the channels in the image data.
    name : str
        The name of the image.
    shape : tuple
        The shape of the image data in the standard SIS format (frames, rows, columns, channels).
    """
    def __init__(self, data, transform, axes, channels, name):
        """
        Parameters
        ----------
        data : xarray.DataArray
            The image data as an xarray DataArray.
        transform : callable
            A transformation function to apply to the image data.
        axes : list of str
            The names of the axes corresponding to the dimensions of the data.
        channels : list of str
            The names of the channels in the image data.
        name : str
            The name of the image.
        """
        super().__init__()
        self._data = data
        self.transform = transform
        self.axes = axes
        self.channels = channels
        self.name = name

    @property
    def shape(self):
        """Returns the shape of the image data in standard SIS order:
        (frames, rows, columns, channels) (z, y, x, c)
        
        Returns
        -------
        tuple
            4D shape (frames, rows, columns, channels)
            
        Raises
        ------
        ValueError
            If the number of dimensions in the data is not between 2 and 4.
        """
        if self._data.ndim > 4 or self._data.ndim < 2:
            raise ValueError('Unsupported shape. ndim must be between 2 and 4')
        axes_dict = dict(zip(self._data.dims, self._data.shape))
        return (axes_dict.get('z', 1), axes_dict.get('y', 1), axes_dict.get('x', 1), axes_dict.get('c', 1))

    def _standard_image_shape(self, img_data):
        """Swizzles the order of the axes so that it is in the standard SIS order
        (frames, rows, columns, channels) (z, y, x, c)
        
        Parameters
        ----------
        img_data : xarray.DataArray
            The image data as an xarray DataArray.
        
        Returns
        -------
        xarray.DataArray
            The image data with axes in standard SIS order.
            
        Raises
        ------
        ValueError
            If the number of dimensions in the data is not between 2 and 4.
        """
        if self._data.ndim > 4 or self._data.ndim < 2:
            raise ValueError('Unsupported shape. ndim must be between 2 and 4')

        # Certain versions of img_data will not have z
        if 'z' not in img_data.dims:
            img_data = img_data.expand_dims('z')

        # Certain versions of img_data will not have c
        if 'c' not in img_data.dims:
            img_data = img_data.expand_dims('c')
            
        return img_data.transpose('z', 'y', 'x', 'c') # Standard SIS order
    
    def get_data(self, channel=None):
        """Return one channel of the image data.
        
        Parameters
        ----------
        channel : str or None, optional
            Name of channel to return data from
                
        Returns
        -------
        np.ndarray
            Subregion of image data (frames, rows, columns)
            
        Raises
        ------
        AssertionError
            If there is more than one channel user must specify which channel to return.
        """
        index = self._get_channel_index(channel)
        return self._standard_image_shape(self._data)[..., index].to_numpy() 
            
    def get_sub_data(self, frames: tuple|None, rows: tuple, cols: tuple, channel: str|None=None):
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
            
        Raises
        ------
        AssertionError
            If there is more than one channel user must specify which channel to return.
        """
        # We repeat some of the above code to keep object as xarray longer. This improves speed
        # If user specified channel or if only one channel, we will return (frames, rows, columns) shaped array
        # There may be a case where a channel is specified without a dimension in the underlying data
        # Then we want to make sure we don't accidentally index it. It will still output 3dimensional array b/c of _standard_image_shape
        if frames is None and self.shape[0] == 1:
            frames = (0, 1)
        elif frames is None and self.shape[0] > 1:
            raise ValueError('frames must be defined if image has more than one frame')
        
        index = self._get_channel_index(channel)
        return self._standard_image_shape(self._data)[frames[0]:frames[1], rows[0]:rows[1], cols[0]:cols[1], index].to_numpy()

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