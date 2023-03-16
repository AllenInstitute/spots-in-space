from __future__ import annotations
import os, json
import numpy as np
from tqdm.notebook import tqdm

import geojson
from scipy.spatial import Delaunay
import shapely
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize

import pandas

from .image import ImageStack
from . import util


class SpotTable:
    """Represents a spatial transcriptomics spot table.
    
    - Contains x, y, z, gene per spot
    - May be a subset of another table, in which case indices are tracked between tables
    - May contain cell IDs loaded from segmentation

    Parameters
    ----------
    pos : array
        Array of shape (N, 3) giving the pos of each detected transcript.
    gene_names : array | None
        Array of shape (N,) giving the name of the gene detected in each transcript. 
        Must specify either *gene_names* or *gene_ids*, not both.
    gene_ids : array | None
        Array of shape (N,) describing the gene detected in each transcript, as an index into *gene_id_to_name*.
        Must specify either *gene_names* or *gene_ids*, not both.
    gene_id_to_name: ndarray | None
        Array mapping from values in *gene_ids* to string names.
    cell_ids : array | None
        Optional array of cell IDs per spot
    parent_table : SpotTable | None
        Indicates that this table is a subset of a parent SpotTable.
    parent_inds : array | None
        Indices used to select the subset of spots in this table from the parent spots.
    parent_region : tuple | None
        X,Y boundaries ((xmin, xmax), (ymin, ymax)) used to select this table from the parent table.
    """
    def __init__(self, 
                 pos: np.ndarray,
                 gene_names: None|np.ndarray=None, 
                 gene_ids: None|np.ndarray=None, 
                 gene_id_to_name: None|np.ndarray=None,
                 cell_ids: None|np.ndarray=None, 
                 parent_table: 'None|SpotTable'=None, 
                 parent_inds: None|np.ndarray=None, 
                 parent_region: None|tuple=None,
                 images: None|list=None,):
        
        self.pos = pos
        self.parent_table = parent_table
        self.parent_inds = parent_inds
        self.parent_region = parent_region

        if gene_names is not None:
            assert gene_ids is None and gene_id_to_name is None
            gene_ids, gene_to_id, id_to_gene = self._make_gene_index(gene_names)
            self.gene_ids = gene_ids
            self.gene_name_to_id = gene_to_id
            self.gene_id_to_name = id_to_gene
        elif gene_ids is not None:
            assert gene_id_to_name is not None
            self.gene_ids = gene_ids
            self.gene_id_to_name = gene_id_to_name
            self.gene_name_to_id = {name:id for id,name in enumerate(self.gene_id_to_name)}
        else:
            raise Exception("Must specify either gene_names or gene_ids")

        self._gene_names = None
        self._cell_ids = cell_ids
        self._cell_index = None
        self._cell_bounds = None
        self.cell_polygons = {}
        
        self.images = []
        if images is not None:
            for img in images:
                self.add_image(img)

    def __len__(self):
        return len(self.pos)

    def dataframe(self, cols=None):
        """Return a dataframe containing the specified columns.

        By default, columns are x, y, z, gene_ids, and cell_ids (if available).
        Also available: gene_names
        """
        if cols is None:
            cols = ['x', 'y', 'z', 'gene_ids']
            if self.pos.shape[1] == 2:
                cols.remove('z')
            if self.cell_ids is not None:
                cols.append('cell_ids')
        return pandas.DataFrame({col:getattr(self, col) for col in cols})

    @property
    def gene_names(self):
        if self._gene_names is None:
            self._gene_names = self.map_gene_ids_to_names(self.gene_ids)
        return self._gene_names

    @property
    def x(self):
        return self.pos[:, 0]

    @property
    def y(self):
        if self.pos.shape[1] < 2:
            return None
        else:
            return self.pos[:, 1]

    @property
    def z(self):
        if self.pos.shape[1] < 3:
            return None
        else:
            return self.pos[:, 2]

    def map_gene_names_to_ids(self, names):
        out = np.empty(len(names), dtype=self.gene_ids.dtype)
        for i,name in enumerate(names):
            out[i] = self.gene_name_to_id[name]
        return out

    def map_gene_ids_to_names(self, ids):
        out = np.empty(len(ids), dtype=self.gene_id_to_name.dtype)
        for i,id in enumerate(ids):
            out[i] = self.gene_id_to_name[id]
        return out

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
        return (self.x.min(), self.x.max()), (self.y.min(), self.y.max())
        
    def cell_ids_changed(self):
        """Call when self.cell_ids has been modified to invalidate caches.
        """
        self._cell_index = None
        self._cell_bounds = None
        
    def get_subregion(self, xlim: tuple, ylim: tuple):
        """Return a SpotTable including the subset of this table inside the region xlim, ylim
        """
        mask = (
            (self.x >= xlim[0]) & 
            (self.x <  xlim[1]) & 
            (self.y >= ylim[0]) & 
            (self.y <  ylim[1])
        )
        sub = self[mask]
        sub.parent_region = (xlim, ylim)
        sub.images = [img.get_subregion(sub.parent_region) for img in sub.images]
        return sub

    def get_genes(self, gene_names=None, gene_ids=None):
        """Return a subtable containing only the genes specified by either gene_names or gene_ids.
        """
        inds = self.gene_indices(gene_names=gene_names, gene_ids=gene_ids)
        return self[inds]

    def save_csv(self, file_name: str, columns: list=None):
        """Save a CSV file with columns x, y, z, gene_id, [gene_name, cell_id].
        
        Optionally, use the *columns* argument to specify which columns to write.
        By default, the cell ID column is only present if cell IDs are available.
        """
        # can't use np.savetext since columns are spread over multiple arrays.
        
        # where to find data for each CSV column
        col_data = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'gene_id': self.gene_ids,
            'cell_id': self.cell_ids,
        }
        if 'gene_name' in columns:
            col_data['gene_name'] = self.gene_names
        
        # how to format each CSV column
        col_fmts = {
            'x': '0.7g',
            'y': '0.7g',
            'z': '0.7g',
            'gene_id': 'd',
            'gene_name': 's',
            'cell_id': 'd',
        }
        
        # which columns to write?
        if columns is None:
            columns = ['x', 'y', 'z', 'gene_id']
            if self.cell_ids is not None:
                columns.append('cell_id')
                
        # write csv
        header = ','.join(columns)
        line_format = ','.join(['{%s:%s}'%(col, col_fmts[col]) for col in columns])
        with open(file_name, 'w') as fh:
            fh.write(header + '\n')
            for i in range(len(self)):
                line = line_format.format(**{col:col_data[col][i] for col in columns})
                fh.write(line)
                fh.write('\n')

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

    @classmethod
    def load_baysor(cls, file_name: str, **kwds):
        """Return a new SpotTable loaded from a baysor result file.
        """
        result = load_baysor_result(file_name, **kwds)
        pos = result[['x', 'y', 'z']].view(dtype=result.dtypes.fields['x'][0]).reshape(len(result), 3)
        return SpotTable(pos=pos, gene_ids=result['gene'], cell_ids=result['cell'])

    @classmethod
    def load_merscope(cls, csv_file: str, cache_file: str|None, image_path: str|None=None, max_rows: int|None=None):
        """Load MERSCOPE data from a detected transcripts CSV file.

        CSV reading is slow, so optionally cache the result to a .npz file.
        """
        if cache_file is None or not os.path.exists(cache_file):
            print("Loading csv..")

            # Which columns are present in csv file?
            cols_in_file = open(csv_file, 'r').readline().split(',')
            cols_in_file = [col.strip() for col in cols_in_file]

            # decide which columns to use for each data source
            col_map = {}
            if 'global_x' in cols_in_file:
                col_map.update({'x': 'global_x', 'y': 'global_y', 'z': 'global_z'})
            else:
                col_map.update({'x': 'x', 'y': 'y', 'z': 'z'})
            col_map['gene'] = 'gene'
            if 'cell_id' in cols_in_file:
                col_map['cell_id'] = 'cell_id'
            col_inds = [cols_in_file.index(c) for c in col_map.values()]

            # pick final dtypes
            dtype = [('x', 'float32'), ('y', 'float32'), ('z', 'float32'), ('gene', 'S20'), ('cell_id', 'int64')]
            dtype = [field for field in dtype if field[0] in col_map]

            # convert positions to 2D array
            raw_data = np.loadtxt(csv_file, skiprows=1, usecols=col_inds, delimiter=',', dtype=dtype, max_rows=max_rows)
            pos = raw_data.view('float32').reshape(len(raw_data), raw_data.itemsize//4)[:, :3]

            # get gene names as fixed-length string
            max_gene_len = max(map(len, raw_data['gene']))
            gene_names = raw_data['gene'].astype(f'U{max_gene_len}')

            # get cell IDs if possible
            cell_ids = None
            if 'cell_id' in col_map:
                cell_ids = raw_data['cell_id']

            # make a spot table!
            table = SpotTable(pos=pos, gene_names=gene_names, cell_ids=cell_ids)

            if cache_file is not None:                
                print("Recompressing to npz..")
                table.save_npz(cache_file)

        else:
            print("Loading from npz..")
            table = cls.load_npz(cache_file)

        # if requested, look for images as well (these are not saved in cache file)
        images = None
        if image_path is not None:
            images = ImageStack.load_merscope_stacks(image_path)
            for img in images:
                table.add_image(img)

        return table

    @classmethod
    def load_stereoseq(cls, gem_file: str|None=None, cache_file: str|None=None, gem_cols: dict|tuple=(('gene', 0), ('x', 1), ('y', 2), ('MIDcounts', 3)), cell_cols: dict|tuple=None, skiprows: int|None=1,  max_rows: int|None=None):
        """
        Load StereoSeq data from gem file. This can be slow so optionally cache the result to a .npz file.
        1/19/2023: New StereoSeq data has cell_ids, add optional cell_cols to add to SpotTable. Also has a flag
        for whether the spot was in the main cell or the extended cell (in_cell)
        """
        if cache_file is None or not os.path.exists(cache_file):
            print('Loading gem...')
            dtype = [('gene', 'S20'), ('x', 'int32'), ('y', 'int32'), ('MIDcounts', 'int')]
            gem_cols = dict(gem_cols)
            usecols = [gem_cols[col] for col in ['gene', 'x', 'y', 'MIDcounts']]
            if cell_cols is not None:
                dtype.extend([('cell_ids', 'int'), ('in_cell', 'bool')])
                usecols.extend([cell_cols[col] for col in ['cell_ids', 'in_cell']])
            fh = open(gem_file, 'r+')
            if skiprows is not None:
                [fh.readline() for i in range(skiprows)]
            pos = []
            gene_ids = []
            gene_name_to_id = {}
            next_gene_id = 0
            end = 0
            file_size = os.stat(gem_file).st_size
            n_rows = 0
            bytes_read = 0
            hit_max_rows = False
            with tqdm(total=file_size) as pbar:
                while end < file_size and not hit_max_rows:
                    start = fh.tell()
                    lines = []
                    for i in range(1000000):
                        if bytes_read >= file_size:
                            break
                        lines.append(fh.readline())
                        n_rows += i  
                        bytes_read += len(lines[-1])
                        if max_rows is not None and max_rows <= n_rows:
                            hit_max_rows = True
                            break
              
                    raw_data = np.loadtxt(lines, usecols=usecols, delimiter='\t', dtype=dtype)
                    end = fh.tell()
                    counts = np.asarray(raw_data['MIDcounts'], dtype='uint8')
                    pos2 = np.empty((sum(counts), 2), dtype='float64')
                    pos2[:, 0] = np.repeat(raw_data['x'], counts)
                    pos2[:, 1] = np.repeat(raw_data['y'], counts)
                    pos.append(pos2)
                    genes = np.repeat(raw_data['gene'], counts)
                    for gene in np.unique(raw_data['gene']):
                        if gene not in gene_name_to_id:
                            gene_name_to_id[gene] = next_gene_id
                            next_gene_id += 1
                    gene_ids2 = np.array([gene_name_to_id[gene] for gene in genes], dtype='uint32')
                    gene_ids.append(gene_ids2)
                    pbar.update(end - start)
                    # if cell_cols is not None:
                    #     cell_ids = np.repeat(raw_data['cell_ids'], counts).astype('uint16')
                    #     in_cell = np.invert(np.repeat(raw_data['in_cell'], counts).astype('bool'))
                        # table = SpotTable(pos=pos, gene_names=genes.astype(f'U{max_gene_len}'), cell_ids=cell_ids)
                        # table.in_cell = in_cell
                    # else:
            gene_ids = np.concatenate(gene_ids)
            pos = np.vstack(pos)
            max_len = max(map(len, gene_name_to_id.keys()))
            gene_id_to_name = np.empty(len(gene_name_to_id), dtype=f'U{max_len}')
            for gene, id in gene_name_to_id.items():
                gene_id_to_name[id] = gene
            table = SpotTable(pos=pos, gene_ids=gene_ids, gene_id_to_name=gene_id_to_name)

            if cache_file is not None:                
                print("Recompressing to npz..")
                table.save_npz(cache_file)

            return table
        else:
            print("Loading from npz..")
            return cls.load_npz(cache_file)

    def save_npz(self, npz_file):
        fields = {
            'pos': self.pos,
            'gene_ids': self.gene_ids, 
            'gene_id_to_name': self.gene_id_to_name,
        }
        if self._cell_ids is not None:
            fields['cell_ids'] = self._cell_ids    
        
        np.savez_compressed(npz_file, **fields) 

    @classmethod
    def load_npz(cls, npz_file):
        fields = np.load(npz_file)
        return SpotTable(**fields)

    def _make_gene_index(cls, gene_names):
        """Given an array of gene names, return an array of integer gene IDs and dictionaries
        that map from gene to ID, and from ID to gene.
        """
        genes = np.unique(gene_names)
        max_len = max(map(len, genes))
        gene_to_id = {}
        id_to_gene = np.empty(len(genes), dtype=f'U{max_len}')

        for i, gene in enumerate(genes):
            gene_to_id[gene] = i
            id_to_gene[i] = gene

        gene_ids = np.array([gene_to_id[gene] for gene in gene_names], dtype=np.min_scalar_type(len(genes)))

        return gene_ids, gene_to_id, id_to_gene

    def filter_cells(self, real_cells=None, min_spot_count=None):
        """Return a filtered spot table containing only cells matching the filter criteria.
        """
        cells, counts = np.unique(self.cell_ids, return_counts=True)

        masks = []

        # filter out spots not associated with cells
        if real_cells is True:
            masks.append(cells > 0)

        # filter for min_count
        if min_spot_count is not None:
            masks.append(counts >= min_spot_count)

        mask = masks[0]
        for m in masks[1:]:
            mask &= m

        cells = cells[m]
        return self[np.isin(self.cell_ids, cells)]

    def cell_by_gene_dataframe(self):
        """Return a pandas dataframe containing a cell-by-gene table derived from this spot table.
        """
        spot_df = pandas.DataFrame({'cell': self.cell_ids, 'gene': self.gene_ids})
        cell_by_gene_df = pandas.pivot_table(spot_df, columns='gene', index='cell', aggfunc=len, fill_value=0)
        cell_by_gene_df.columns = self.map_gene_ids_to_names(cell_by_gene_df.columns)
        return cell_by_gene_df

    def cell_bounds(self, cell_id: int):
        """Return xmin, xmax, ymin, ymax for *cell_id*
        """
        if self._cell_bounds is None:
            self._cell_bounds = {}
            for cid in np.unique(self.cell_ids):
                inds = self.cell_indices(cid)
                rows = self.pos[inds]
                self._cell_bounds[cid] = (
                    rows[:,0].min(),
                    rows[:,0].max(),
                    rows[:,1].min(),
                    rows[:,1].max(),
                )
        return self._cell_bounds[cell_id]

    @staticmethod
    def cell_polygon(cell_points_array, alpha_inv):
        """
        get 2D alpha shape from a set of points, modified slightly from 
        http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

        args:
        cell_points  numpy array of  point locations with expected columns [x,y].
        alpha_inv parameter that sets the radius filter on the Delaunay triangulation.  
                traditionally alpha is defined as 1/radius, 
                and here the function input is inverted for slightly more intuitive use
        """
        tri = Delaunay(cell_points_array)
        # Make a list of line segments: 
        # edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
        #                 ((x1_2, y1_2), (x2_2, y2_2)),
        #                 ... ]

        edge_points = []
        edges = set()

        def add_edge(i, j):
            """Add a line between the i-th and j-th points, if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add( (i, j) )
            edge_points.append(cell_points_array[ [i, j] ])

        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = cell_points_array[ia]
            pb = cell_points_array[ib]
            pc = cell_points_array[ic]

            # Lengths of sides of triangle
            a = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = np.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = np.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

            # Semiperimeter of triangle
            s = (a + b + c)/2.0

            # Area of triangle by Heron's formula
            area = np.sqrt(s*(s-a)*(s-b)*(s-c))

            circum_r = a*b*c/(4.0*area)

            # Here's the radius filter.
            if circum_r < alpha_inv:
                add_edge(ia, ib)
                add_edge(ib, ic)
                add_edge(ic, ia)

        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        tp = unary_union(triangles)
        
        return tp

    @staticmethod
    def calculate_cell_features( cell_polygon):
        return  {"area":cell_polygon.area, "centroid":np.array(cell_polygon.centroid.coords)}

    def calculate_cell_polygons(self, alpha_inv=1.5):
        # run through all cell_ids, generate polygons and add to self.cell_polygons dict 
        # increases the alpha_inv parameter by 0.5 until a single polygon is generated
        self.cell_polygons = {}
        for cid in tqdm(np.unique(self.cell_ids)):
            inds = self.cell_indices(cid)
            xy_pos = self.pos[inds][:, :2]
            if xy_pos.shape[0] > 3:
                putative_polygon = self.cell_polygon(xy_pos, alpha_inv)
                # increase alpha_inv unti we only have 1 polygon... this should be pretty rare.
                tries = 0
                flex_alpha_inv = alpha_inv
                while tries <20 and isinstance(putative_polygon, shapely.geometry.MultiPolygon  ):
                    flex_alpha_inv += .5
                    tries += 1
                    putative_polygon = self.cell_polygon(xy_pos, flex_alpha_inv)

                self.cell_polygons[cid] = putative_polygon
            else:    
                self.cell_polygons[cid] = None

    def get_cell_features(self):
        # run through self.polys and calculate features
        if len(self.cell_polygons.keys()) == 0:
            return None

        cell_features = []
        for cid in self.cell_polygons:
            feature_info = {"cell_id":cid}
            if self.cell_polygons[cid]:
                feature_info.update(self.calculate_cell_features(self.cell_polygons[cid]))
            else:
                feature_info.update(dict(area=0., centroid = None))
            cell_features.append(feature_info)

        return pandas.DataFrame.from_records(cell_features)

    def save_cell_polygons(self, geojson_save_path):
        """
        save a geojson geometry collection from the cell polygons
        """

        if len(self.cell_polygons)==0:
            return
        
        geojsonROIs = []
        for poly_key in self.cell_polygons.keys():
            if self.cell_polygons[poly_key]:
                if len(self.cell_polygons[poly_key].exterior.coords) > 2:
                    geojsonROIs.append(util.poly_to_geojson(self.cell_polygons[poly_key]))
        
        with open(geojson_save_path, 'w') as w:
            json.dump( geojson.GeometryCollection(geojsonROIs), w)

        
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
            if len(cell_ids) == 0:
                return np.array([], dtype=int)
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

    def gene_indices(self, gene_names=None, gene_ids=None):
        """Return an array of indices where the detected transcript is in either gene_names or gene_ids
        """
        assert (gene_names is not None) != (gene_ids is not None)
        if gene_names is not None:
            gene_ids = self.map_gene_names_to_ids(gene_names)
        return np.argwhere(np.isin(self.gene_ids, gene_ids))[:, 0]
    
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
        pos = self.pos[item]
        gene_ids = self.gene_ids[item]
        cell_ids = None if self.cell_ids is None else self.cell_ids[item]

        if len(pos) > 0:
            parent_region = ((pos[:,0].min(), pos[:,0].max()), (pos[:,1].min(), pos[:,1].max()))
        else:
            parent_region = None

        subset = type(self)(
            pos=pos,
            gene_ids=gene_ids,
            gene_id_to_name=self.gene_id_to_name,
            cell_ids=cell_ids, 
            parent_table=self, 
            parent_inds=np.arange(len(self))[item],
            parent_region=parent_region,
        )

        subset.images = self.images[:]
            
        return subset
    
    def copy(self, deep:bool=False, **kwds):
        """Return a copy of self, optionally with some attributes replaced.
        """
        init_kwargs = dict(
            parent_table=self.parent_table,
            parent_inds=self.parent_inds,
            parent_region=self.parent_region,
        )
        init_kwargs.update(kwds)
        for name in ['pos', 'gene_ids', 'gene_id_to_name', 'cell_ids', 'images']:
            if name not in init_kwargs:
                val = getattr(self, name)
                if deep:
                    val = None if val is None else val.copy()
                init_kwargs[name] = val
            
        return SpotTable(**init_kwargs)

    def split_tiles(self, max_spots_per_tile: int=None, target_tile_width: float=None, overlap: float=30):
        """Return a list of SpotTables that tile this one.

        This table will be split into rows of equal height, and each row will be split into
        columns with roughly the same number of spots (less than *max_spots_per_tile*).
        
        see also: grid_tiles

        Parameters
        ----------
        max_spots_per_tile : int | None
            Maximum number of spots to include in each tile.
        target_tile_width : float | None
            Automatically select max_spots_per_tile to get an approximate tile width
        overlap : float
            Distance to overlap tiles

        """
        assert (max_spots_per_tile == None) != (target_tile_width == None), "Must specify either max_spots_per_tile or target_tile_width"

        if max_spots_per_tile is None:
            # convert target_tile_width to max_spots_per_tile
            x_range = self.bounds()[0][1] - self.bounds()[0][0]
            target_n_cols = x_range / target_tile_width
            max_spots_per_tile = len(self) / target_n_cols**2
            max_spots_per_tile *= 1.2 * (target_tile_width + overlap/2)**2 / target_tile_width**2

        max_spots_per_tile = int(max_spots_per_tile)
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
            row_table = self.get_subregion(
                xlim=bounds[0],
                ylim=(max(bounds[1][0], start_y - padding), min(bounds[1][1], stop_y + padding)))
            row_bounds = row_table.bounds()

            # sort x values
            order = np.argsort(row_table.x)
            xvals = row_table.x[order]

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
                    padded_start_x = max(xvals[0], xvals[start] - padding)
                    padded_stop_x = min(xvals[-1], xvals[stop] + padding)
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

    def grid_tiles(self, max_tile_size:float, overlap: float=30):
        """Return a grid of overlapping tiles with equal size, where the width and height
        must be less than max_tile_size.

        See also: split_tiles
        """
        assert max_tile_size > 2 * overlap
        bounds = self.bounds()
        width = bounds[0][1] - bounds[0][0]
        height = bounds[1][1] - bounds[1][0]

        n_cols = int(width / max_tile_size)
        while True:
            n_cols += 1
            col_width = (width + (n_cols - 1) * overlap) / n_cols
            if col_width <= max_tile_size:
                break

        n_rows = int(height / max_tile_size)
        while True:
            n_rows += 1
            row_height = (height + (n_rows - 1) * overlap) / n_rows
            if row_height <= max_tile_size:
                break

        tiles = []
        for row in tqdm(range(n_rows)):
            ystart = bounds[1][0] + row * (row_height - overlap)
            ylim = (ystart, ystart + row_height)
            row_tile = self.get_subregion(bounds[0], ylim)
            for col in range(n_cols):
                xstart = bounds[0][0] + col * (col_width - overlap)
                xlim = (xstart, xstart + col_width)
                tile = row_tile.get_subregion(xlim, ylim)
                if len(tile) > 0:
                    tiles.append(tile)
                # re-parent tile to self rather than row_table
                tile.parent_table = self
                tile.parent_inds = row_tile.map_indices_to_parent(tile.parent_inds)
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
        """
        # copy *other* because we will modify cell IDs
        other = other.copy(cell_ids=other.cell_ids.astype(int, copy=True))

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

    def set_cell_ids_from_tiles(self, tiles, padding=5):
        """Overwrite all cell IDs by merging from *tiles*, which may be a list of SpotTable
        or SegmentationResult instances.
        """
        from .segmentation import SegmentationResult
        # create empty cell ID table (where -1 means nothing has been assigned yet)
        self.cell_ids = np.empty(len(self), dtype=int)
        self.cell_ids[:] = -1
        merge_results = []
        for tile in tqdm(tiles):
            if isinstance(tile, SegmentationResult):
                tile = tile.spot_table()            
            result = self.merge_cells(tile.copy(), padding=5)
            merge_results.append(result)
        return merge_results

    def plot_rect(self, ax, color):
        """Plot the bounding region of this table as a rectangle.
        """
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
        """Generate a color palette suitable for distinguisging individual cells
        """
        import seaborn
        cell_set = np.unique(cells)
        colors = seaborn.color_palette('tab20b', 30)
        palette = {cid: colors[i%len(colors)] for i, cid in enumerate(cell_set)}
        palette[0] = (0.5, 0.5, 0.5, 0.05)
        palette[-1] = (0.5, 0.5, 0.5, 0.05)
        palette[-2] = (1, 0, 1)
        palette[-3] = (0, 1, 0)
        palette[-4] = (0, 0, 1)
        palette[-5] = (1, 0, 0)
        return palette

    def scatter_plot(self, ax=None, x='x', y='y', color='gene_ids', alpha=0.2, size=1.5, z_slice=None):
        import seaborn
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        
        if z_slice is not None:
            zvals = np.unique(self.z)
            zval = zvals[int(z_slice * (len(zvals)-1))]
            mask = self.z == zval
            self = self[mask]
            
        if color == 'cell': 
            palette = self.cell_palette(self.cell_ids)
            color = 'cell_ids'
        else:
            palette = None

        seaborn.scatterplot(
            data=self.dataframe(cols=[x, y, color]),
            x=x, 
            y=y, 
            hue=color, 
            palette=palette,
            linewidth=0, 
            alpha=alpha,
            s=size,
            ax=ax,
            legend=False
        )
        ax.set_aspect('equal')

    def cell_scatter_plot(self, *args, **kwds):
        """Scatter plot of spots colored by cell ID
        """
        kwds['color'] = 'cell'
        return self.scatter_plot(*args, **kwds)

    def binned_expression_counts(self, binsize):
        """Return an array of spatially binned gene expression counts with shape (n_x_bins, n_y_bins, n_genes)
        """
        x = self.x
        y = self.y
        gene = self.gene_ids
        
        xrange = x.min(), x.max()
        yrange = y.min(), y.max()
        
        gene_id_bins = np.arange(len(self.gene_id_to_name) + 1)
        x_bins = int(np.ceil((xrange[1] - xrange[0]) / binsize))
        y_bins = int(np.ceil((yrange[1] - yrange[0]) / binsize))
        
        hist = np.histogramdd(
            sample=np.stack([x, y, gene], axis=1),
            bins=(x_bins, y_bins, gene_id_bins),
        )
                    
        return hist
    
    def reduced_expression_map(self, binsize, umap_args=None, ax=None, umap_ax=None, norm=util.log_plus_1):
        import seaborn
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if umap_args is None:
            umap_args = dict(
                min_dist=0.4,
                n_neighbors=10,
                random_state=0,
            )

        print("Binning expression counts..")
        bec, (xbins, ybins, gbins) = self.binned_expression_counts(binsize=binsize)

        print("Reducing binned expression counts..")
        umap_args['n_components'] = 2
        if norm is not None:
            norm_bec = norm(bec)
        else:
            norm_bec = bec
        reduced = util.reduce_expression(norm_bec, umap_args=umap_args)
        
        norm = bec.sum(axis=2)
        norm = norm / norm.max()
        color = util.rainbow_wheel(reduced) * np.sqrt(norm[:, :, None])
        
        xrange = xbins[0], xbins[-1]
        yrange = ybins[0], ybins[-1]
        util.show_float_rgb(color.transpose(1, 0, 2), extent=xrange + yrange, ax=ax)

        if umap_ax is not None:
            flat = reduced.reshape(reduced.shape[0] * reduced.shape[1], reduced.shape[2])
            color = util.rainbow_wheel(flat)
            seaborn.scatterplot(x=flat[:,0], y=flat[:,1], c=color, alpha=0.2, ax=umap_ax)

        return bec, (xbins, ybins, gbins), (reduced,)

    def show_image(self, ax, image_size=300, log=True):
        """Show an image of binned spot positions.
        """
        xbins = np.linspace(self.x.min(), self.x.max(), image_size)
        ybins = np.linspace(self.y.min(), self.y.max(), image_size)
        hist = np.histogram2d(self.x, self.y, bins=[xbins, ybins])
        if log:
            img = np.log(hist[0] + 1)
        else:
            img = hist[0]
        ax.imshow(img.T, origin='lower', aspect='equal', cmap='inferno', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])

    def show_subregion_images(self, subtables, ax, **kwds):
        """Show a spot table image and successive subregions
        """
        tables = [self] + list(subtables)
        for i, table in enumerate(tables):
            table.show_image(ax=ax[i], **kwds)
            if i > 0:
                table.plot_rect(ax[i-1], 'c')

    def add_image(self, image):
        """Attach an image to this dataset
        """
        if image.name is not None and image.name in [img.name for img in self.images]:
            raise Exception(f"An image named {image.name} is already attached")
        self.images.append(image)

    def get_image(self, name=None, channel=None):
        """Return the image with the given name or channel name
        """
        if name is not None:
            selected = [img for img in self.images if img.name == name]
            if len(selected) == 0:
                raise Exception(f"No image found with name={name}")
        elif channel is not None:
            selected = [img for img in self.images if channel in img.channels]
            if len(selected) == 0:
                raise Exception(f"No image found with channel {channel}")
            if len(selected) > 1:
                raise Exception(f"Multiple images found with channel {channel}")
        else:
            raise Exception("Must specify at least one of name or channel")
        selected_img = selected[0]
            
        if channel is not None:
            return selected_img.get_channel(channel)
        else:
            return selected_img            
        
    def show_image(self, ax, channel=None, z_index=None, z_pos=None, name=None):
        """Show a channel / z plane from an image
        """
        img = self.get_image(name=name, channel=channel)
        if z_index is not None:
            img = img.get_z_index(z_index)
        if z_pos is not None:
            img = img.get_z_pos(z_pos)
            
        return img
        


