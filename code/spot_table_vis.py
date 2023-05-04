"""Tool for visualizing spot tables when interactivity is needed (for example, to QC segmentation)

Usage: python -i spot_table_vis.py merscope_detected_transcripts.npz segmentation.npy

Needs an environment with vispy; try:

    conda create --name spotvis python=3 numpy scipy pandas h5py jupyter scikit-image scikit-learn matplotlib seaborn
    conda activate spotvis
    pip install pyqt5 vispy
"""
import sys
import sawg
import numpy as np
import vispy.scene
from vispy.scene import visuals
import matplotlib.pyplot as plt


table_file = sys.argv[1]
table = sawg.SpotTable.load_merscope(csv_file=None, cache_file=table_file)

if len(sys.argv) > 2:
    segmentation_file = sys.argv[2]
    table.cell_ids = np.load(segmentation_file)

# filter for cells with a minimum spot count
# everything else will be visible as a dim cyan
filtered = table.filter_cells(real_cells=True, min_spot_count=200)
filtered_ids = np.unique(filtered.cell_ids)

# subset spots if this requires too much memory for your system
subset = table[::3]


canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
vb = canvas.central_widget.add_view()
vb.camera = 'panzoom'
vb.camera.rect = (-10, -10, 20, 20)
vb.camera.aspect = 1

scatter = visuals.Markers(scaling=True)
scatter.set_gl_state('translucent', depth_test=False)
vb.add(scatter)

color = plt.cm.tab20b(subset.cell_ids % 20)
color[:, 3] = 0.4

# spots assigned to filtered cells are dim cyan
mask = np.isin(subset.cell_ids, filtered_ids)
color[~mask] = (0, 1, 1, 0.2)

# spots not assigned to a cell a grey
color[subset.cell_ids < 1] = (1, 1, 1, 0.1)

scatter.set_data(subset.pos, edge_width=0, face_color=color, size=1)
vb.camera.set_range()
