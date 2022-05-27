import os, json
import numpy as np
from tqdm.notebook import tqdm


class SpotTable:
    """Represents a spatial transcriptomics spot table.
    
    - Contains x, y, z, gene per spot
    - May be a subset of another table, in which case indices are tracked between tables
    - May contain cell IDs loaded from segmentation

    Parameters
    ----------
    data : array
        Structured array of spots with columns x, y, z, and gene (at least)
    cell_ids : array | None
        Optional array of cell IDs per spot
    parent_table : SpotTable | None
        Indicates that this table is a subset of a parent SpotTable.
    parent_inds : array | None
        Indices used to select the subset of spots in this table from the parent spots.
    parent_region : tuple | None
        X,Y boundaries ((xmin, xmax), (ymin, ymax)) used to select this table from the parent table.
        
    """
    def __init__(self, data: np.ndarray, cell_ids: None|np.ndarray=None, parent_table: 'None|SpotTable'=None, parent_inds: None|np.ndarray=None, parent_region: None|tuple=None):
        self.data = data
        self.parent_table = parent_table
        self.parent_inds = parent_inds
        self.parent_region = parent_region
        self._cell_ids = cell_ids
        self._cell_index = None
        self._cell_bounds = None
        
    def __len__(self):
        return len(self.data)
        
    @property
    def cell_ids(self):
        """An array of cell IDs.
        """
        return self._cell_ids
    
    @cell_ids.setter
    def cell_ids(self, cid: np.ndarray):
        self._cell_ids = cid
        self.cell_ids_changed()
        
    def bounds(self):
        """Return ((xmin, xmax), (ymin, ymax)) giving the boundaries of data included in this table.
        """
        return (self.data['x'].min(), self.data['x'].max()), (self.data['y'].min(), self.data['y'].max())
        
    def cell_ids_changed(self):
        """Call when self.cell_ids has been modified to invalidate caches.
        """
        self._cell_index = None
        self._cell_bounds = None
        
    def get_subregion(self, xlim: tuple, ylim: tuple):
        """Return a SpotTable including the subset of this table inside the region xlim, ylim
        """
        mask = (
            (self.data['x'] >= xlim[0]) & 
            (self.data['x'] <  xlim[1]) & 
            (self.data['y'] >= ylim[0]) & 
            (self.data['y'] <  ylim[1])
        )
        return self[mask]

    def save_csv(self, file_name: str):
        """Save a CSV file with columns x, y, z, gene that is appropriate for running Baysor.
        """
        args = dict(
            fmt=['%0.7g', '%0.7g', '%0.7g', '%d'],
            header='x,y,z,gene',
            delimiter=',',
            comments='',
        )
        np.savetxt(file_name, self.data, **args)

    def save_json(self, json_file: str):
        """Save parent_region and parent_inds to json.
        """
        json_data = {
            'parent_region': [(float(rgn[0]), float(rgn[1])) for rgn in self.parent_region],
            'parent_inds': [int(i) for i in self.parent_inds],
        }
        json.dump(json_data, open(json_file, 'w'))

    def load_json(self, json_file: str):
        """Return a subset of this table as described by a json file (previously saved with save_json)
        """
        json_data = json.load(open(json_file, 'r'))
        sub_table = self[json_data['parent_inds']]
        sub_table.parent_region = json_data['parent_region']
        return sub_table

    def load_cell_ids(self, file_name: str):
        """Load cell IDs from a baysor segmentation and assign them to self.cell_ids.
        """
        self._baysor_result = load_baysor_result(file_name, remove_noise=False, remove_no_cell=False)
        assert len(self._baysor_result) == len(self)
        self.cell_ids = self._baysor_result['cell']

    @classmethod
    def load_baysor(cls, file_name: str, **kwds):
        """Return a new SpotTable loaded from a baysor result file.
        """
        result = load_baysor_result(file_name, **kwds)
        return SpotTable(data=result, cell_ids=result['cell'])
        
    def cell_bounds(self, cell_id: int):
        """Return xmin, xmax, ymin, ymax for *cell_id*
        """
        if self._cell_bounds is None:
            self._cell_bounds = {}
            for cid in np.unique(self.cell_ids):
                inds = self.cell_indices(cid)
                rows = self.data[inds]
                self._cell_bounds[cid] = (
                    rows['x'].min(),
                    rows['x'].max(),
                    rows['y'].min(),
                    rows['y'].max(),
                )
        return self._cell_bounds[cell_id]
        
    def cell_indices(self, cell_ids: int | np.ndarray):
        """Return indices giving table location of all spots with *cell_ids*
        """
        if self._cell_index is None:
            ind = {}
            for i, cid in enumerate(self.cell_ids):
                ind.setdefault(cid, []).append(i)
            self._cell_index = ind

        if isinstance(cell_ids, (int, np.integer)):
            return self._cell_index[cell_ids]
        else:
            return np.concatenate([self._cell_index[cid] for cid in cell_ids])

    def cells_inside_region(self, xlim: tuple, ylim: tuple):
        """Return IDs of cells that are entirely inside xlim, ylim.
        """
        cell_ids = []
        for cid in np.unique(self.cell_ids):
            x0, x1, y0, y1 = self.cell_bounds(cid)
            if x0 > xlim[0] and x1 < xlim[1] and y0 > ylim[0] and y1 < ylim[1]:
                cell_ids.append(cid)
        return cell_ids

    def cell_mask(self, cell_ids: int | np.ndarray):
        """Return a mask selecting spots that belong to cells in *cell_ids*
        """
        mask = np.zeros(len(self), dtype=bool)
        for cid in cell_ids:
            mask[self.cell_indices(cid)] = True
        return mask
    
    def map_indices_to_parent(self, inds: np.ndarray):
        """Given an array of indices into this SpotTable, return a new array of indices that
        select the same spots in the parent SpotTable.
        """
        return self.parent_inds[inds]
    
    def map_indices_from_parent(self, inds: np.ndarray):
        """Given an array of indices into the parent SpotTable, return a new array of indices that
        select the same spots in this SpotTable.
        """
        inv_map = {b:a for a,b in enumerate(self.parent_inds)}
        return np.array([inv_map[i] for i in inds])
    
    def map_mask_to_parent(self, mask: np.ndarray):
        """Given a boolean mask that selects spots from this SpotTable, return a new boolean mask
        that selects the same spots from the parent SpotTable.
        """
        parent_mask = np.zeros(len(self.parent_table), dtype=bool)
        parent_mask[self.parent_inds] = mask
        return parent_mask
    
    def __getitem__(self, item: int|np.ndarray):
        """Return a subset of this SpotTable.

        *item* may be an integer array of indices to select, or a boolean mask array.
        """
        data = self.data[item]
        subset = type(self)(
            data=data, 
            parent_table=self, 
            parent_inds=np.arange(len(self))[item],
            parent_region=((data['x'].min(), data['x'].max()), (data['y'].min(), data['y'].max()))
        )
        
        cells = self.cell_ids
        if cells is not None:
            subset.cell_ids = cells[item]
            
        return subset
    
    def copy(self, deep:bool=False, **kwds):
        """Return a copy of self, optionally with some attributes replaced.
        """
        defaults = dict(
            parent_table=self.parent_table,
            parent_inds=self.parent_inds,
            parent_region=self.parent_region,
        )
        defaults.update(kwds)
        if 'data' not in defaults:
            defaults['data'] = self.data.copy() if deep else self.data
        if 'cell_ids' not in defaults:
            defaults['cell_ids'] = self.cell_ids.copy() if deep else self.cell_ids
            
        return SpotTable(**defaults)

    def split_tiles(self, max_spots_per_tile: int, overlap: float):
        """Return a list of SpotTables that tile this one.

        This table will be split into rows of equal height, and each row will be split into
        columns with roughly the same number of spots (less than *max_spots_per_tile*).

        Parameters
        ----------
        max_spots_per_tile : int
            Maximum number of spots to include in each tile.
        overlap : float
            Distance to overlap tiles

        """
        padding = overlap / 2
        est_n_tiles = len(self) // max_spots_per_tile
        bounds = self.bounds()
        tot_width, tot_height = (bounds[0][1] - bounds[0][0]), (bounds[1][1] - bounds[1][0])
        aspect_ratio = tot_width / tot_height
        est_n_cols = est_n_tiles ** 0.5 * aspect_ratio
        n_rows = int(np.ceil(est_n_tiles / est_n_cols))
        tile_height = tot_height / n_rows

        tiles = []
        for row in tqdm(range(n_rows)):
            # get a subregion for the entire row
            start_y = bounds[1][0] + row * tile_height
            stop_y = start_y + tile_height
            row_table = self.get_subregion(xlim=bounds[0], ylim=(start_y - padding, stop_y + padding))
            row_bounds = row_table.bounds()

            # sort x values
            order = np.argsort(row_table.data['x'])
            xvals = row_table.data['x'][order]

            # split into columns
            n_cols = 1 + len(row_table) // max_spots_per_tile
            while True:
                # iteratively attempt smaller columns until everything fits
                init_spots_per_col = 1 + len(row_table) // n_cols
                cols = []
                failed = False
                for col in range(n_cols):
                    # start with equal chunks of data
                    start = col * init_spots_per_col
                    stop = min(start + init_spots_per_col, len(row_table) - 1)

                    # adjust start / stop to account for padding
                    padded_start_x = xvals[start] - padding
                    padded_stop_x = xvals[stop] + padding
                    start_adj = np.searchsorted(xvals, padded_start_x)
                    stop_adj = np.searchsorted(xvals, padded_stop_x)

                    # create the tile
                    tile_inds = order[start_adj:stop_adj]
                    tile = row_table[tile_inds]

                    # re-parent tile to self rather than row_table
                    tile.parent_region = ((padded_start_x, padded_stop_x), row_bounds[1])
                    tile.parent_table = self
                    tile.parent_inds = row_table.map_indices_to_parent(tile_inds)

                    # if this tile is too big, then start the row over with smaller tiles
                    if len(tile) > max_spots_per_tile:
                        failed = True
                        break
                    cols.append(tile)
                if failed:
                    n_cols += 1
                    continue
                else:
                    break
            tiles.extend(cols)
        return tiles

    def cell_indices_within_padding(self, padding=5.0):
        """Return spot indices all cells that do not come within *padding* of the parent_region.

        This is used to exclude cells near the edge of a tile, where the segmentation becomes unreliable.
        """
        tile_xlim, tile_ylim = self.parent_region
        include_xlim = (tile_xlim[0] + padding, tile_xlim[1] - padding) 
        include_ylim = (tile_ylim[0] + padding, tile_ylim[1] - padding) 
        include_cells = self.cells_inside_region(xlim=include_xlim, ylim=include_ylim)
        include_inds = self.cell_indices(include_cells)
        return include_inds

    def merge_cells(self, other, padding=5):
        """Merge cell IDs from SpotTable *other* into self.

        Returns a structure describing merge conflicts.

        Note: this method modifies the cell IDs in *other* in-place!
        """
        # increment cell IDs in new tile (leaving cell 0 unchanged)
        other.cell_ids[other.cell_ids > 0] += self.cell_ids.max()
        other.cell_ids_changed()

        # get indices of all cells that are not close to the edge of the tile partial cells from edge of tile
        tile_inds = other.cell_indices_within_padding(padding=padding)
        self_inds = other.map_indices_to_parent(tile_inds)

        # keep track of state before merge so we can look for conflicts afterward
        original_state = self[other.parent_inds]

        # copy all retained cells from first tile to full table
        self.cell_ids[self_inds] = other.cell_ids[tile_inds]
        self.cell_ids_changed()

        # At this point the merge is done
        # ------------------------------- 
        # the rest is just collecting information about possible merge conflicts
        new_state = self[other.parent_inds]

        # cells in original state that were affected by merge
        affected_cell_ids = set(original_state.cell_ids[tile_inds]) - set([-1, 0])

        # all cells present in new merge area
        new_cell_ids = set(new_state.cell_ids) - set([-1, 0])

        # set of cells that were partially replaced by the merge
        # (cells that were affected by the merge, but not completely replaced by the merge)
        partial_merge_cells = new_cell_ids & affected_cell_ids

        conflicts = []
        for cell_id in partial_merge_cells:
            old_inds = original_state.cell_indices(cell_id)
            new_inds = new_state.cell_indices(cell_id)

            # which new cells replaced part of the old cell?
            overlapped_cells = set(new_state.cell_ids[old_inds]) - set([-1, 0])
            overlapped_cell_inds = {cid:new_state.cell_indices(cid) for cid in overlapped_cells}

            # how big are the overlapping cells?
            original_cell_size = len(old_inds)
            reduced_cell_size = len(new_inds)
            overlapped_cell_sizes = {cid:len(inds) for cid, inds in overlapped_cell_inds.items()}
            overlapped_cell_parent_inds = {cid:new_state.map_indices_to_parent(inds) for cid, inds in overlapped_cell_inds.items()}

            conflicts.append({
                'original_cell_id': cell_id,
                'original_size': original_cell_size,
                'size_after_merge': reduced_cell_size,
                'size_ratio': reduced_cell_size / original_cell_size,
                'original_indices': new_state.map_indices_to_parent(old_inds),
                'overlapped_cells': overlapped_cells,
                'overlapped_cell_sizes': overlapped_cell_sizes,
                'overlapped_cell_indices': overlapped_cell_parent_inds,
            })

        return conflicts

    def plot_rect(self, ax, color):
        import matplotlib.patches
        xlim, ylim = self.parent_region
        pos = (xlim[0], ylim[0])
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        rect = matplotlib.patches.Rectangle(pos, width, height, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        return rect
    
    @staticmethod
    def cell_palette(cells):
        cell_set = np.unique(cells)
        n_cells = len(cell_set)
        colors = seaborn.color_palette('dark', 30)
        palette = {cid: colors[i%len(colors)] for i, cid in enumerate(cell_set)}
        palette[0] = (0, 1, 1)
        palette[-1] = (1, 1, 0)
        palette[-2] = (1, 0, 1)
        palette[-3] = (0, 1, 0)
        palette[-4] = (0, 0, 1)
        palette[-5] = (1, 0, 0)
        return palette

    def cell_scatter_plot(self, ax, alpha=0.2, size=1.5, z_slice=None):
        import seaborn
        if z_slice is not None:
            zvals = np.unique(self.data['z'])
            zval = zvals[int(z_slice * (len(zvals)-1))]
            mask = self.data['z'] == zval
            self = self[mask]
        
        seaborn.scatterplot(
            x=self.data['x'], 
            y=self.data['y'], 
            hue=self.cell_ids, 
            palette=self.cell_palette(self.cell_ids), 
            linewidth=0, 
            alpha=alpha,
            size=size,
            ax=ax,
            legend=False
        )
        ax.set_aspect('equal')


def load_baysor_result(result_file, remove_noise=True, remove_no_cell=True):
    dtype = [('x', 'float32'), ('y', 'float32'), ('z', 'float32'), ('cluster', int), ('cell', int), ('is_noise', bool)]
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
    

