sis.spot_table
==============

SpotTable
---------

.. autoclass:: sis.spot_table.SpotTable
    :show-inheritance:
    :class-doc-from: class
    :no-members:
   
.. rubric:: Initialization
.. autosummary::
    :toctree: generated/

	sis.SpotTable.__init__
	sis.SpotTable.load_merscope
	sis.SpotTable.load_stereoseq
	sis.SpotTable.load_xenium
	sis.SpotTable.load_npz
	sis.SpotTable.load_pickle
	sis.SpotTable.load_merscope_spatialdata
	sis.SpotTable.load_xenium_spatialdata

.. rubric:: Properties
.. autosummary::
    :toctree: generated/

    sis.SpotTable.x
    sis.SpotTable.y
    sis.SpotTable.z
    sis.SpotTable.gene_names

.. rubric:: Writing and saving
.. autosummary::
    :toctree: generated/

    sis.SpotTable.save_csv
    sis.SpotTable.save_npz
    sis.SpotTable.dataframe
    
.. rubric:: Plotting
.. autosummary::   
    :toctree: generated/
    
    sis.SpotTable.scatter_plot
    sis.SpotTable.show_image
    sis.SpotTable.plot_rect
    sis.SpotTable.reduced_expression_map
    sis.SpotTable.show_binned_heatmap
    

.. rubric:: Methods
.. autosummary::
    :toctree: generated/

    sis.SpotTable.__len__
    sis.SpotTable.__getitem__
    sis.SpotTable.copy
    sis.SpotTable.bounds
    sis.SpotTable.get_subregion
    sis.SpotTable.map_gene_names_to_ids
    sis.SpotTable.map_gene_ids_to_names
    sis.SpotTable.get_genes
    sis.SpotTable.detect_z_planes
    sis.SpotTable.z_plane_mask
    sis.SpotTable.gene_indices
    sis.SpotTable.map_indices_to_parent
    sis.SpotTable.map_indices_from_parent
    sis.SpotTable.map_mask_to_parent
    sis.SpotTable.split_tiles
    sis.SpotTable.grid_tiles
    sis.SpotTable.add_image
    sis.SpotTable.get_image
    sis.SpotTable.save_json
    sis.SpotTable.load_json
    sis.SpotTable.binned_expression_counts

SegmentedSpotTable
------------------

.. autoclass:: sis.spot_table.SegmentedSpotTable
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization & Loading
.. autosummary::
    :toctree: generated/

	sis.SegmentedSpotTable.__init__
	sis.SegmentedSpotTable.load_npz
	sis.SegmentedSpotTable.load_merscope
	sis.SegmentedSpotTable.load_xenium
	sis.SegmentedSpotTable.load_stereoseq
	sis.SegmentedSpotTable.load_merscope_spatialdata
	sis.SegmentedSpotTable.load_xenium_spatialdata
	sis.SegmentedSpotTable.load_cell_polygons

.. rubric:: Properties
.. autosummary::
    :toctree: generated/

	sis.SegmentedSpotTable.cell_ids
	sis.SegmentedSpotTable.cell_labels
	sis.SegmentedSpotTable.unique_cell_ids

.. rubric:: Writing and saving
.. autosummary::
    :toctree: generated/

	sis.SegmentedSpotTable.save_npz
    sis.SegmentedSpotTable.dataframe
	sis.SegmentedSpotTable.cell_by_gene_dense_matrix
	sis.SegmentedSpotTable.cell_by_gene_sparse_matrix
	sis.SegmentedSpotTable.cell_by_gene_anndata
	sis.SegmentedSpotTable.save_anndata
	sis.SegmentedSpotTable.get_geojson_collection
	sis.SegmentedSpotTable.save_cell_polygons
    sis.SegmentedSpotTable.save_xenium_kit_cbg
    
.. rubric:: Plotting
.. autosummary::   
    :toctree: generated/
    
	sis.SegmentedSpotTable.scatter_plot
	sis.SegmentedSpotTable.cell_scatter_plot
	sis.SegmentedSpotTable.cell_palette
    

.. rubric:: Methods
.. autosummary::
    :toctree: generated/

	sis.SegmentedSpotTable.__len__
	sis.SegmentedSpotTable.__getattr__
	sis.SegmentedSpotTable.__getitem__
	sis.SegmentedSpotTable.copy
	sis.SegmentedSpotTable.get_subregion
	sis.SegmentedSpotTable.generate_cell_labels
    sis.SegmentedSpotTable.convert_cell_id
	sis.SegmentedSpotTable.convert_cell_label
	sis.SegmentedSpotTable.cell_indices
	sis.SegmentedSpotTable.cell_mask
	sis.SegmentedSpotTable.filter_cells
	sis.SegmentedSpotTable.cells_inside_region
	sis.SegmentedSpotTable.cell_centroids
    sis.SegmentedSpotTable.cell_bounds
	sis.SegmentedSpotTable.calculate_cell_polygons
	sis.SegmentedSpotTable.calculate_all_cell_features
