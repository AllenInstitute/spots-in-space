sis.image
=========

ImageBase
---------

.. autoclass:: sis.image.ImageBase
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Properties
.. autoproperty:: sis.image.ImageBase.shape

.. rubric:: Plotting
.. automethod:: sis.image.ImageBase.show

.. rubric:: Methods
.. autosummary::   
    :toctree: generated/
    
    sis.image.ImageBase.bounds
    sis.image.ImageBase.get_channel
    sis.image.ImageBase.get_frame
    sis.image.ImageBase.get_frames
    sis.image.ImageBase.get_subregion
    sis.image.ImageBase.get_data
    sis.image.ImageBase.get_sub_data

Image
-----

.. autoclass:: sis.image.Image
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.Image.__init__
    
.. rubric:: Properties
.. autoproperty:: sis.image.Image.shape

.. rubric:: Methods
.. automethod:: sis.image.Image.get_data
.. automethod:: sis.image.Image.get_sub_data

ImageFile
---------

.. autoclass:: sis.image::ImageFile
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Properties
.. autoproperty:: sis.image.ImageFile.shape

.. rubric:: Methods
.. automethod:: sis.image.ImageFile.get_data
.. automethod:: sis.image.ImageFile.get_sub_data

MerscopeImageFile
-----------------

.. autoclass:: sis.image.MerscopeImageFile
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.MerscopeImageFile.__init__
.. automethod:: sis.image.MerscopeImageFile.load

.. rubric:: Properties
.. autoproperty:: sis.image.MerscopeImageFile.shape

.. rubric:: Methods
.. automethod:: sis.image.MerscopeImageFile.get_data
.. automethod:: sis.image.MerscopeImageFile.get_sub_data

StereoSeqImageFile
------------------

.. autoclass:: sis.image.StereoSeqImageFile
    :show-inheritance:
    :class-doc-from: class
    :no-members:
  
.. rubric:: Initialization
.. automethod:: sis.image.StereoSeqImageFile.__init__
.. automethod:: sis.image.StereoSeqImageFile.load
 
.. rubric:: Properties
.. autoproperty:: sis.image.StereoSeqImageFile.shape

.. rubric:: Methods
.. automethod:: sis.image.StereoSeqImageFile.get_data
.. automethod:: sis.image.StereoSeqImageFile.get_sub_data

XeniumImageFile
---------------

.. autoclass:: sis.image.XeniumImageFile
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.XeniumImageFile.__init__
.. automethod:: sis.image.XeniumImageFile.load

.. rubric:: Properties
.. autoproperty:: sis.image.XeniumImageFile.shape

.. rubric:: Methods
.. automethod:: sis.image.XeniumImageFile.get_data
.. automethod:: sis.image.XeniumImageFile.get_sub_data

ImageTransform
--------------

.. autoclass:: sis.image.ImageTransform
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.ImageTransform.__init__
.. automethod:: sis.image.ImageTransform.load_spatialdata_transformation

.. rubric:: Properties
.. autoproperty:: sis.image.ImageTransform.inverse_matrix

.. rubric:: Methods
.. automethod:: sis.image.ImageTransform.map_to_pixels
.. automethod:: sis.image.ImageTransform.map_from_pixels
.. automethod:: sis.image.ImageTransform.translated    

ImageStack
----------

.. autoclass:: sis.image.ImageStack
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.ImageStack.__init__
.. automethod:: sis.image.ImageStack.load_merscope_stacks
.. automethod:: sis.image.ImageStack.load_spatialdata_stacks

.. rubric:: Properties
.. autoproperty:: sis.image.ImageStack.shape
.. autoproperty:: sis.image.ImageStack.channels
.. autoproperty:: sis.image.ImageStack.transform

.. rubric:: Plotting
.. automethod:: sis.image.ImageStack.show
   
.. rubric:: Methods
.. autosummary::   
    :toctree: generated/

    sis.image.ImageStack.bounds
    sis.image.ImageStack.get_channel
    sis.image.ImageStack.get_frame
    sis.image.ImageStack.get_frames
    sis.image.ImageStack.get_subregion
    sis.image.ImageStack.get_data
    sis.image.ImageStack.get_sub_data

ImageView
---------

.. autoclass:: sis.image.ImageView
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.ImageView.__init__

.. rubric:: Properties
.. autoproperty:: sis.image.ImageView.name
.. autoproperty:: sis.image.ImageView.shape
.. autoproperty:: sis.image.ImageView.channels

.. rubric:: Plotting
.. automethod:: sis.image.ImageView.show

.. rubric:: Methods
.. autosummary::   
    :toctree: generated/
    
    sis.image.ImageView.bounds
    sis.image.ImageView.get_channel
    sis.image.ImageView.get_frame
    sis.image.ImageView.get_frames
    sis.image.ImageView.get_subregion
    sis.image.ImageView.get_data
    sis.image.ImageView.get_sub_data

SpatialDataImage
----------------

.. autoclass:: sis.image.SpatialDataImage
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.image.SpatialDataImage.__init__

.. rubric:: Properties
.. autoproperty:: sis.image.SpatialDataImage.shape

.. rubric:: Plotting
.. automethod:: sis.image.ImageBase.show

.. rubric:: Methods
.. autosummary::   
    :toctree: generated/
    
    sis.image.SpatialDataImage.bounds
    sis.image.SpatialDataImage.get_channel
    sis.image.SpatialDataImage.get_frame
    sis.image.SpatialDataImage.get_frames
    sis.image.SpatialDataImage.get_subregion
    sis.image.SpatialDataImage.get_data
    sis.image.SpatialDataImage.get_sub_data