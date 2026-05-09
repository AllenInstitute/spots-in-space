sis.segmentation
================

SegmentationMethod
------------------

.. autoclass:: sis.segmentation.SegmentationMethod
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.segmentation.SegmentationMethod.__init__

.. rubric:: Methods
.. automethod:: sis.segmentation.SegmentationMethod.run

CellposeSegmentationMethod
--------------------------

.. autoclass:: sis.segmentation.CellposeSegmentationMethod
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.segmentation.CellposeSegmentationMethod.__init__

.. rubric:: Methods
.. automethod:: sis.segmentation.CellposeSegmentationMethod.run
.. automethod:: sis.segmentation.CellposeSegmentationMethod.get_total_mrna_image
.. automethod:: sis.segmentation.CellposeSegmentationMethod.map_spots_to_img_px

SegmentationResult
------------------

.. autoclass:: sis.segmentation.SegmentationResult
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.segmentation.SegmentationResult.__init__

.. rubric:: Properties
.. autoproperty:: sis.segmentation.SegmentationResult.cell_ids

.. rubric:: Writing and saving
.. automethod:: sis.segmentation.SegmentationResult.save

.. rubric:: Methods
.. automethod:: sis.segmentation.SegmentationResult.spot_table

CellposeSegmentationResult
--------------------------

.. autoclass:: sis.segmentation.CellposeSegmentationResult
    :show-inheritance:
    :class-doc-from: class
    :no-members:

.. rubric:: Initialization
.. automethod:: sis.segmentation.CellposeSegmentationResult.__init__

.. rubric:: Properties
.. autoproperty:: sis.segmentation.CellposeSegmentationResult.cell_ids
.. autoproperty:: sis.segmentation.CellposeSegmentationResult.mask_image

.. rubric:: Writing and saving
.. automethod:: sis.segmentation.CellposeSegmentationResult.save

.. rubric:: Methods
.. automethod:: sis.segmentation.CellposeSegmentationResult.spot_table

SegmentationPipeline
--------------------

.. autoclass:: sis.segmentation.SegmentationPipeline
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization & Running
.. automethod:: sis.segmentation.SegmentationPipeline.__init__
.. automethod:: sis.segmentation.SegmentationPipeline.from_json
.. automethod:: sis.segmentation.SegmentationPipeline.run

.. rubric:: run() Steps
.. autosummary::
    :toctree: generated/

	sis.segmentation.SegmentationPipeline.update_metadata
	sis.segmentation.SegmentationPipeline.save_metadata
	sis.segmentation.SegmentationPipeline.load_raw_spot_table
	sis.segmentation.SegmentationPipeline.tile_seg_region
    sis.segmentation.SegmentationPipeline.get_seg_run_spec
    sis.segmentation.SegmentationPipeline.submit_jobs
    sis.segmentation.SegmentationPipeline.merge_segmented_tiles
	sis.segmentation.SegmentationPipeline.get_polygon_run_spec
	sis.segmentation.SegmentationPipeline.merge_cell_polygons
    sis.segmentation.SegmentationPipeline.create_cell_by_gene
	sis.segmentation.SegmentationPipeline.save_seg_spot_table
	sis.segmentation.SegmentationPipeline.clean_up

.. rubric:: Support functions
.. autosummary::
    :toctree: generated/

    sis.segmentation.SegmentationPipeline.load_metadata
	sis.segmentation.SegmentationPipeline.load_regions
	sis.segmentation.SegmentationPipeline.load_cell_ids
	sis.segmentation.SegmentationPipeline.load_seg_spot_table
	sis.segmentation.SegmentationPipeline.load_cbg
	sis.segmentation.SegmentationPipeline.track_job_progress
    sis.segmentation.SegmentationPipeline.rerun_failed_jobs
	sis.segmentation.SegmentationPipeline.find_failed_jobs
	sis.segmentation.SegmentationPipeline.resubmit_failed_jobs
	sis.segmentation.SegmentationPipeline.update_jobs

MerscopeSegmentationPipeline
----------------------------

.. autoclass:: sis.segmentation.MerscopeSegmentationPipeline
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.segmentation.MerscopeSegmentationPipeline.__init__

StereoSeqSegmentationPipeline
-----------------------------

.. autoclass:: sis.segmentation.StereoSeqSegmentationPipeline
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.segmentation.StereoSeqSegmentationPipeline.__init__

XeniumSegmentationPipeline
--------------------------

.. autoclass:: sis.segmentation.XeniumSegmentationPipeline
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.segmentation.XeniumSegmentationPipeline.__init__