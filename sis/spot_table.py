from __future__ import annotations
import os, json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
import pandas

from .optional_import import optional_import
tqdm = optional_import('tqdm.notebook', names=['tqdm'])[0]
geojson = optional_import('geojson')
shapely = optional_import('shapely')
MultiLineString = optional_import('shapely.geometry', names=['MultiLineString'])[0]
unary_union, polygonize = optional_import('shapely.ops', ['unary_union', 'polygonize'])

from .image import ImageBase, ImageFile, ImageStack, ImageTransform
from . import util
from . import _version

def run_cell_polygon_calculation(load_func, load_args:dict, cell_id_file: str|None, subregion: str|tuple|None, cell_subset_file:str|None, result_file:str|None, alpha_inv_coeff: float=1, separate_z_planes=True):
    """Load a segmented spot table, calculate the cell polygons (possibly on a subset of cells), and save the result.
    """
    if cell_id_file is not None:
        # Load a raw SpotTable and add cell ids
        # If using a subregion, the cell_id_file should only include cell_ids for that subregion
        print('Loading SpotTable and cell ids...', end='')
        spot_table = load_func(**load_args)
        if subregion is not None:
            subregion = spot_table.get_image(channel=subregion).bounds() if isinstance(subregion, str) else subregion
            spot_table = spot_table.get_subregion(*subregion)
        cell_ids = np.load(cell_id_file)
        seg_spot_table = SegmentedSpotTable(spot_table, cell_ids)

    else:
        # Load a SegmentedSpotTable with cell_ids included
        print('Loading SegmentedSpotTable...', end='')
        seg_spot_table = load_func(**load_args)
        if subregion is not None:
            subregion = seg_spot_table.get_image(channel=subregion).bounds() if isinstance(subregion, str) else subregion
            seg_spot_table = seg_spot_table.get_subregion(*subregion)

    print(f"subregion {subregion} {len(seg_spot_table)}")
    print('[DONE]')

    if cell_subset_file is not None:
        cells_to_run = np.load(cell_subset_file)
    else:
        cells_to_run = np.unique(seg_spot_table.cell_ids)
        cells_to_run = np.delete(cells_to_run, np.where((cells_to_run == 0) | (cells_to_run == -1)))

    print('Calculating Cell Polygons...')
    seg_spot_table.calculate_cell_polygons(cells_to_run=cells_to_run, alpha_inv_coeff=alpha_inv_coeff, separate_z_planes=separate_z_planes)

    print('Saving Cell Polygons...', end='')
    seg_spot_table.save_cell_polygons(result_file)
    print('[DONE]')


class SpotTable:
    """Represents a spatial transcriptomics spot table.
    
    - Contains x, y, z, gene per spot
    - May be a subset of another table, in which case indices are tracked between tables
    - If your table is segmented (e.g. contains cell_ids), use SegmentedSpotTable

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
    parent_table : SpotTable | None
        Indicates that this table is a subset of a parent SpotTable.
    parent_inds : array | None
        Indices used to select the subset of spots in this table from the parent spots.
    parent_region : tuple | None
        X,Y boundaries ((xmin, xmax), (ymin, ymax)) used to select this table from the parent table.
    images : list[ImageBase] | ImageBase | None
        Image(s) associated with the data (e.g. nuclei stain).
    """
    def __init__(self, 
                 pos: np.ndarray,
                 gene_names: None|np.ndarray=None, 
                 gene_ids: None|np.ndarray=None, 
                 gene_id_to_name: None|np.ndarray=None,
                 parent_table: 'None|SpotTable'=None, 
                 parent_inds: None|np.ndarray=None, 
                 parent_region: None|tuple=None,
                 images: None|list[ImageBase]|ImageBase=None,
                 ):
 
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

        self.images = []
        if images is None:
            pass
        elif isinstance(images, ImageBase):
            self.add_image(images)
        elif isinstance(images, list) and all([isinstance(i, ImageBase) for i in images]):
            for img in images:
                self.add_image(img)
        else:
            raise TypeError(f'Unsupported type for image(s). Images must be of type ImageBase or a list of ImageBase.')

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, item: np.ndarray):
        """Return a subset of this SpotTable.

        *item* may be an integer array of indices to select, or a boolean mask array.
        """
        pos = self.pos[item]
        gene_ids = self.gene_ids[item]

        if len(pos) > 0:
            parent_region = ((pos[:,0].min(), pos[:,0].max()), (pos[:,1].min(), pos[:,1].max()))
        else:
            parent_region = None

        subset = type(self)(
            pos=pos,
            gene_ids=gene_ids,
            gene_id_to_name=self.gene_id_to_name,
            parent_table=self, 
            parent_inds=np.arange(len(self))[item],
            parent_region=parent_region,
        )

        subset.images = self.images[:]
            
        return subset
    
    def dataframe(self, cols=['x', 'y', 'z', 'gene_ids']):
        """Return a dataframe containing the specified columns.

        By default, columns are x, y, z, gene_ids.
        Also available: gene_names
        """
        if self.pos.shape[1] == 2 and 'z' in cols:
            cols.remove('z')
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

        
    def bounds(self):
        """Return ((xmin, xmax), (ymin, ymax)) giving the boundaries of data included in this table.
        """
        return (self.x.min(), self.x.max()), (self.y.min(), self.y.max())
        
    def get_subregion(self, xlim: tuple, ylim: tuple, incl_end: bool=False):
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
        sub.images = [img.get_subregion(sub.parent_region, incl_end=incl_end) for img in sub.images]
        return sub

    def get_genes(self, gene_names=None, gene_ids=None):
        """Return a subtable containing only the genes specified by either gene_names or gene_ids.
        """
        inds = self.gene_indices(gene_names=gene_names, gene_ids=gene_ids)
        return self[inds]


    def detect_z_planes(self, float_cut: float|None=None):
        """Return a tuple containing (min z-plane, max z-plane + 1).
        
        Parameters
        ----------
        float_cut : float | None
            If specified, only include z-planes that contain at least this fraction of spots.
        """
        if float_cut:
            z_planes, z_counts = np.unique(self.z, return_counts=True)
            z_counts = z_counts / np.sum(z_counts)
            z_planes = z_planes[z_counts >= float_cut]
        else:
            z_planes = np.unique(self.z)

        return int(np.min(z_planes)), int(np.max(z_planes)) + 1


    def z_plane_mask(self, z_planes: tuple):
        """Return a mask of the SpotTable containing only spots in the specified z_planes
        Z-planes are specified as a tuple [min, max) to align with the output of detect_z_planes.
        
        Input:
            z_planes: tuple containing the min (inclusive) and max (exclusive) z_planes to keep
        """
        return np.isin(self.z, [z for z in range(*z_planes)])


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
    def load_pickle(cls, file_name: str):
        """Return a new SpotTable loaded a SpotTable pickle stored on disk
        """
        import pickle

        with open(file_name, 'rb') as f:
            table = pickle.load(f)
        return table

    @staticmethod
    def read_merscope_csv(csv_file, cols_to_use=('x', 'y', 'z', 'gene'), max_rows=None):
        """Helper function to read a MERSCOPE csv file.
        Intended to reduce duplicated code between SpotTable.load_merscope()
        and SegmentedSpotTable.load_merscope().
        """

        # Which columns are present in csv file?
        with open(csv_file, 'r') as f:
            cols_in_file = f.readline().split(',')
        cols_in_file = [col.strip() for col in cols_in_file]
        
        # decide which columns to use for each data source
        col_map = {col: col for col in cols_to_use}
        if 'global_x' in cols_in_file:
            col_map.update({'x': 'global_x', 'y': 'global_y', 'z': 'global_z'})

        col_inds = [cols_in_file.index(c) for c in col_map.values()]

        # pick final dtypes
        dtype = [('x', 'float32'), ('y', 'float32'), ('z', 'float32'), ('gene', 'S20')]
        dtype = [field for field in dtype if field[0] in col_map]
        
        # convert positions to 2D array
        raw_data = np.loadtxt(csv_file, skiprows=1, usecols=col_inds, delimiter=',', dtype=dtype, max_rows=max_rows)
        pos = raw_data.view('float32').reshape(len(raw_data), raw_data.itemsize//4)[:, :3]

        # get gene names as fixed-length string
        max_gene_len = max(map(len, raw_data['gene']))
        gene_names = raw_data['gene'].astype(f'U{max_gene_len}')

        return raw_data, pos, gene_names

    @classmethod
    def load_merscope(cls, csv_file: str, cache_file: str|None, image_path: str|None=None, max_rows: int|None=None):
        """Load MERSCOPE data from a detected transcripts CSV file. This is the
        preferred method for resegmentation. If you want the original MERSCOPE
        segmentation, use SegmentedSpotTable.load_merscope.

        Parameters
        ----------
        csv_file : str
            Path to the detected transcripts file.
        cache_file : str, optional
            Path to the detected transcripts cache file, which is an npz file 
            representing the raw SpotTable (without cell_ids). If passed, will
            create a cache file if one does not already exists.
        image_path : str, optional
            Path to directory containing a MERSCOPE image stack.
        max_rows : int, optional
            Maximum number of rows to load from the CSV file.

        Returns
        -------
        sis.spot_table.SpotTable
        """
        # if requested, look for images (these are not saved in cache file)
        images = None
        if image_path is not None:
            images = ImageStack.load_merscope_stacks(image_path)

        if cache_file is None or not os.path.exists(cache_file):
            print("Loading csv..")

            raw_data, pos, gene_names = SpotTable.read_merscope_csv(csv_file=csv_file, max_rows=max_rows)

            if cache_file is not None:                
                print("Recompressing to npz..")
                cls(pos=pos, gene_names=gene_names, images=images).save_npz(cache_file)

            return cls(pos=pos, gene_names=gene_names, images=images)

        else:
            print("Loading from npz..")
            return cls.load_npz(cache_file, images=images)

    @staticmethod
    def read_stereoseq_gem(gem_file: str, gem_cols: dict|tuple, cell_col: int|None, skiprows: int|None, max_rows: int|None):
        """Helper function to read raw data from a StereoSeq dataset."""
        xyscale = 0.5  # um per unit (stereoseq spots are separated by 500 nm)

        dtype = [('gene', 'S20'), ('x', 'int32'), ('y', 'int32'), ('MIDcounts', 'int')]
        gem_cols = dict(gem_cols)
        usecols = [gem_cols[col] for col in ['gene', 'x', 'y', 'MIDcounts']]
        if cell_col is not None:
            dtype.append(('cell_ids', 'int'))
            usecols.append(cell_col) 
        fh = open(gem_file, 'r+')
        if skiprows is not None:
            [fh.readline() for i in range(skiprows)]
        pos = []
        gene_ids = []
        cell_ids = []
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
                        extra_rows = n_rows - max_rows
                        lines = lines[:-extra_rows]
                        break
          
                raw_data = np.loadtxt(lines, usecols=usecols, delimiter='\t', dtype=dtype)
                end = fh.tell()
                counts = np.asarray(raw_data['MIDcounts'], dtype='uint8')
                pos2 = np.empty((sum(counts), 2), dtype='float64')
                pos2[:, 0] = np.repeat(raw_data['x'] * xyscale, counts)
                pos2[:, 1] = np.repeat(raw_data['y'] * xyscale, counts)
                pos.append(pos2)
                genes = np.repeat(raw_data['gene'], counts)
                for gene in np.unique(raw_data['gene']):
                    if gene not in gene_name_to_id:
                        gene_name_to_id[gene] = next_gene_id
                        next_gene_id += 1
                gene_ids2 = np.array([gene_name_to_id[gene] for gene in genes], dtype='uint32')
                gene_ids.append(gene_ids2)
                pbar.update(end - start)
                if cell_col is not None:
                    cell_ids.append(np.repeat(raw_data['cell_ids'], counts).astype('uint16'))
        gene_ids = np.concatenate(gene_ids)
        pos = np.vstack(pos)
        max_len = max(map(len, gene_name_to_id.keys()))
        gene_id_to_name = np.empty(len(gene_name_to_id), dtype=f'U{max_len}')
        for gene, id in gene_name_to_id.items():
            gene_id_to_name[id] = gene

        if cell_col is not None:
            return pos, gene_ids, gene_id_to_name, cell_ids
        
        else: 
            return pos, gene_ids, gene_id_to_name


    @classmethod
    def load_stereoseq(cls, gem_file: str|None=None, cache_file: str|None=None, gem_cols: dict|tuple=(('gene', 0), ('x', 1), ('y', 2), ('MIDcounts', 3)), 
                       cell_col: int|None=None, skiprows: int|None=1,  max_rows: int|None=None, image_file: str|None=None, image_channel: str|None=None):
        """
        Load StereoSeq data from gem file. This can be slow so optionally cache the result to a .npz file.
        1/19/2023: New StereoSeq data has cell_ids, add optional cell_cols to add to SpotTable. Also has a flag
        for whether the spot was in the main cell or the extended cell (in_cell)
        1/3/2024: new cellbin.gem doesn't have `in_cell`, let's not worry about it and just specify the column 
        that the cell ID is in
        """
        xyscale = 0.5
        assert cell_col == None

        if image_file is not None:
            transform = ImageTransform(matrix=np.array([
                [0, 1/xyscale, 0],
                [1/xyscale, 0, 0],
            ], dtype=float))
            img = ImageFile(file=image_file, transform=transform, axes=['frame', 'row', 'col', 'channel'], channels=[image_channel], name="mosaic")
        else:
            img = None

        if cache_file is None or not os.path.exists(cache_file):
            print('Loading gem...')
            pos, gene_ids, gene_id_to_name = SpotTable.read_stereoseq_gem(gem_file, gem_cols, cell_col, skiprows, max_rows)
                
            if cache_file is not None:                
                print("Recompressing to npz..")
                cls(pos=pos, gene_ids=gene_ids, gene_id_to_name=gene_id_to_name, images=img).save_npz(cache_file)

            return cls(pos=pos, gene_ids=gene_ids, gene_id_to_name=gene_id_to_name, images=img)

        else:
            print("Loading from npz..")
            return cls.load_npz(cache_file, img)


    @staticmethod
    def read_xenium_transcripts(transcript_file, max_rows=None, z_depth: float=3.0):
        """Helper function to read a Xenium transcripts file. (currently supports only csv and parquet)
        Intended to reduce duplicated code between SpotTable.load_xenium()
        and SegmentedSpotTable.load_xenium().
        """
        
        if str(transcript_file).endswith('.csv') or str(transcript_file).endswith('.csv.gz'):
            spot_dataframe = pandas.read_csv(transcript_file, nrows=max_rows)
        elif str(transcript_file).endswith('.parquet'):
            if max_rows:
                raise ValueError('max_rows is not supported for parquet files as pandas does not allow partial reading')
            spot_dataframe = pandas.read_parquet(transcript_file, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported file type for transcript_file: {transcript_file}")

        pos= spot_dataframe.loc[:,["x_location","y_location","z_location"]].values
        z_bins = np.arange(0, np.max(pos[:, 2]) + z_depth, z_depth) # bin float z locations to integers
        pos[:, 2] = (np.digitize(pos[:,2], z_bins) - 1).astype(int)

        gene_names = spot_dataframe.loc[:,"feature_name"].values

        return pos, gene_names
    
    
    @classmethod
    def load_xenium(cls, transcript_file: str, cache_file: str, image_path: str=None, max_rows: int=None, z_depth: float=3.0, keep_images_in_memory: bool=True):
        """ Load Xenium data from a detected transcripts CSV file.
            This is the preferred method for resegmentation. If you want the original Xenium
            segmentation, use SegmentedSpotTable.load_xenium.
            CSV reading is slow, so optionally cache the result to a .npz file.

            Parameters
            ----------
            transcript_file : str
                Path to the detected transcripts file.
            cache_file : str, optional
                Path to the detected transcripts cache file, which is an npz file 
                representing the raw SpotTable (without cell_ids). If passed, will
                create a cache file if one does not already exists.
            image_path : str, optional
                Path to directory containing a Xenium image stack.
            max_rows : int, optional
                Maximum number of rows to load from the CSV file.
            z_depth : float, optional
                Depth (in um) of a imaging layer i.e. z-plane
                Used to bin z-positiions into discrete planes
            keep_images_in_memory : bool
                Xenium images are large and not memory mapped and thus we may want to keep them in memory or not.
                The trade off is speed vs memory.

            Returns
            -------
            sis.spot_table.SpotTable
        """
        # if requested, look for images as well (these are not saved in cache file)
        images = None
        if image_path is not None:
            images = ImageStack.load_xenium_stacks(image_path, keep_images_in_memory=keep_images_in_memory)

        if (cache_file is None) or (not Path(cache_file).exists()):
            print("Loading transcripts...")
            pos, gene_names = SpotTable.read_xenium_transcripts(transcript_file=transcript_file, max_rows=max_rows, z_depth=z_depth)

            if cache_file is not None:                
                print("Recompressing to npz..")
                cls(pos=pos, gene_names=gene_names, images=images).save_npz(cache_file)

            return cls(pos=pos, gene_names=gene_names, images=images)

        else:
            print("Loading from npz..")
            return cls.load_npz(cache_file, images=images)


    def save_npz(self, npz_file: str):
        """Save this SpotTable as an NPZ file."""
        fields = {
            'pos': self.pos,
            'gene_ids': self.gene_ids,
            'gene_id_to_name': self.gene_id_to_name
        }
        
        np.savez_compressed(npz_file, **fields) 

    @classmethod
    def load_npz(cls, npz_file: str, images: ImageBase|list[ImageBase]|None=None):
        """Load from an NPZ file.

        Parameters
        ----------
        npz_file : str
            Path to the npz file.
        images : ImageBase or list of ImageBase, optional
            Image(s) to attach to the SpotTable. Must be loaded separately
            since these cannot be stored in the NPZ file.

        Returns
        -------
        sis.spot_table.SpotTable
        """
        fields = np.load(npz_file)

        # only taking specific fields enables creation of raw spot tables from SegmentedSpotTable npz cache files
        pos = fields['pos']
        gene_ids = fields['gene_ids'] 
        gene_id_to_name = fields['gene_id_to_name']
        return cls(pos=pos, gene_ids=gene_ids, gene_id_to_name=gene_id_to_name, images=images)

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
    
    def copy(self, deep:bool=False, **kwds):
        """Return a copy of self, optionally with some attributes replaced.
        """
        init_kwargs = dict(
            parent_table=self.parent_table,
            parent_inds=self.parent_inds,
            parent_region=self.parent_region,
        )
        init_kwargs.update(kwds)
        for name in ['pos', 'gene_ids', 'gene_id_to_name', 'images',]:
            if name not in init_kwargs:
                val = getattr(self, name)
                if deep:
                    val = None if val is None else val.copy()
                init_kwargs[name] = val
            
        return SpotTable(**init_kwargs)

    def split_tiles(self, max_spots_per_tile: int=None, target_tile_width: float=None, overlap: float=30, incl_end: bool=False, min_transcripts=0):
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
        incl_end : bool
            Include all pixels of the image that overlap with the region, rather than just those inside the region.
        min_transcripts : int
            Minimum number of transcripts in a tile to be returned

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
                ylim=(max(bounds[1][0], start_y - padding), min(bounds[1][1], stop_y + padding)),
                incl_end=incl_end)
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
            
        filter_tiles = [tile for tile in tiles if len(tile) >= min_transcripts]
            
        return filter_tiles

    def grid_tiles(self, max_tile_size:float, overlap: float=30, incl_end: bool=False, min_transcripts=0):
        """Return a grid of overlapping tiles with equal size, where the width and height
        must be less than max_tile_size.

        max_tile_size : int | None
            Maximum width and height of each tile.
        overlap : float
            Distance to overlap tiles
        incl_end : bool
            Include all pixels of the image that overlap with the region, rather than just those inside the region.
        min_transcripts : int
            Minimum number of transcripts in a tile to be returned

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
            row_tile = self.get_subregion(bounds[0], ylim, incl_end=incl_end)
            for col in range(n_cols):
                xstart = bounds[0][0] + col * (col_width - overlap)
                xlim = (xstart, min(xstart + col_width, bounds[0][1]))
                tile = row_tile.get_subregion(xlim, ylim, incl_end=incl_end)
                if len(tile) > 0:
                    tiles.append(tile)
                # re-parent tile to self rather than row_table
                tile.parent_table = self
                tile.parent_inds = row_tile.map_indices_to_parent(tile.parent_inds)
                
        filter_tiles = [tile for tile in tiles if len(tile) >= min_transcripts]
            
        return filter_tiles


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
    
    def scatter_plot(self, ax=None, x='x', y='y', color='gene_ids', alpha=0.2, size=1.5, z_slice=None, palette=None):
        import seaborn
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        
        if z_slice is not None:
            zvals = np.unique(self.z)
            zval = zvals[int(z_slice * (len(zvals)-1))]
            mask = self.z == zval
            self = self[mask]

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

    def show_binned_heatmap(self, ax, image_size=300, log=True):
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

    def get_image(self, name=None, channel=None, frame:int|None=None, frames:tuple|None=None):
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
            selected_img = selected_img.get_channel(channel)

        if frames is not None:
            selected_img = selected_img.get_frames(frames)
        elif frame is not None:
            selected_img = selected_img.get_frame(frame)

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



class SegmentedSpotTable:
    """Represents a collection of cells for a spatial transcriptomics dataset
    that has been segmented.

    - Contains a SpotTable
    - Must contain cell_ids loaded from segmentation
    - May contain cell_polygons
    - May contain segmentation metadata

    Parameters
    ----------
    spot_table : SpotTable
        The spot table associated with the dataset, containing information 
        about transcript locations and gene ids.
    cell_ids : numpy.ndarray
        Array of integer cell ids per spot.
    production_cell_ids : numpy.ndarray, optional
        Array of string production cell ids per spot.
    pcid_to_cid : dict, optional
        Mapping of production cell ids to cell ids.
    cid_to_pcid : dict, optional
        Mapping of cell ids to production cell ids.
    cell_polygons : dict, optional
        Polygons associated with each cell_id. Used to approximate the shapes
        of cells and measurements such as volume.
    seg_metadata : dict, optional
        Metadata about segmentation, e.g. method, parameters, options. Will be
        saved in the cell by gene anndata uns if created.
    """

    def __init__(
            self,
            spot_table: SpotTable,
            cell_ids: np.ndarray,
            production_cell_ids: None|np.ndarray=None,
            pcid_to_cid: None|dict=None,
            cid_to_pcid: None|dict=None,
            cell_polygons: None|dict=None,
            seg_metadata: None|dict=None
            ):
                
        if len(cell_ids) != len(spot_table):
            raise ValueError(f"Number of cell_ids {len(cell_ids)} does not match number of spots {len(spot_table)}")

        self.spot_table = spot_table
        self._old_cell_ids = None
        self._cell_ids = cell_ids
        self._production_cell_ids = production_cell_ids
        self._pcid_to_cid = pcid_to_cid
        self._cid_to_pcid = cid_to_pcid
        self._cell_index = None
        self._cell_bounds = None
        self._unique_cell_ids = None
        self.cell_polygons = cell_polygons
        self.seg_metadata = seg_metadata

    def __len__(self):
        return len(self.cell_ids)

    def __getattr__(self, name):
        """This method enables SegmentedSpotTable to act as a proxy for
        attributes in SpotTable. If the user attempts to look up an attribute
        that is not found in SegmentedSpotTable, this method will try to look
        for this attribute in the attached SpotTable object instead."""

        try:
            attr = getattr(self.spot_table, name)
        except AttributeError as e:
            # add additional info to the error message before raising
            e.args += (f'{name} is not available in the SegmentedSpotTable object or the attached SpotTable.',)
            raise

        return attr

    def __getitem__(self, item: np.ndarray):
        """Return a subset of this SegmentedSpotTable.

        *item* may be an integer array of indices to select, or a boolean mask array.
        """
        spot_table = self.spot_table[item]
        cell_ids = self.cell_ids[item]
        production_cell_ids = None if self.production_cell_ids is None else self.production_cell_ids[item]
         # cell_polygons is not converted because we cannot guarantee the polygons will stay the same after subsetting

        subset = type(self)(
                spot_table=spot_table,
                cell_ids=cell_ids, 
                production_cell_ids=production_cell_ids, 
                pcid_to_cid=self._pcid_to_cid,
                cid_to_pcid=self._cid_to_pcid,
        )
            
        return subset

    @property
    def cell_ids(self):
        """An array of integer cell IDs.
        """
        return self._cell_ids
    
    @cell_ids.setter
    def cell_ids(self, cid: np.ndarray):
        self._old_cell_ids = self._cell_ids.copy()
        self._cell_ids = cid
        self.cell_ids_changed()

    @property
    def production_cell_ids(self):
        """An array of string production cell IDs.
        """
        return self._production_cell_ids
    
    @production_cell_ids.setter
    def production_cell_ids(self, pcids: np.ndarray):
        # Make sure we can map cell_ids to production_cell_ids without conflict and vice versa
        test_df = pandas.DataFrame(self.cell_ids, index=np.arange(len(self.cell_ids)), columns=['cell_ids'])
        test_df['production_cell_ids'] = pcids
        test_df = test_df.drop_duplicates(['cell_ids', 'production_cell_ids'])
        exists_cid_to_pcid_mapping = test_df.groupby('cell_ids')['production_cell_ids'].count().max() == 1
        exists_pcid_to_cid_mapping = test_df.groupby('production_cell_ids')['cell_ids'].count().max() == 1
        if exists_cid_to_pcid_mapping and exists_pcid_to_cid_mapping:
            self._pcid_to_cid = dict(zip(test_df['production_cell_ids'], test_df['cell_ids']))
            self._cid_to_pcid = dict(zip(test_df['cell_ids'], test_df['production_cell_ids']))
            self._production_cell_ids = pcids
        else:
            raise ValueError("There must exist a valid function mapping cell_ids to production_cell_ids and vice versa.")


    def convert_cell_id(self, cell_id: int|str):
        """Convert a cell id to a production cell id and vice versa.

        Parameters
        ----------
        cell_id : int or str
            The cell id (type int) or production cell id (type str) to query.

        Returns
        -------
        str or int
            The other version of the queried cell id.
        """
        if self.production_cell_ids is None:
            raise ValueError("production_cell_ids must be set to use convert_cell_id()")
        if isinstance(cell_id, (int, np.integer)):
            return self._cid_to_pcid[cell_id]
        elif isinstance(cell_id, str):
            return self._pcid_to_cid[cell_id]
        else:
            raise ValueError("cell_id must be of type 'int' or 'str'")

    def cell_ids_changed(self):
        """Call when self.cell_ids has been modified to invalidate caches.
        """
        import warnings
        self._cell_index = None
        self._cell_bounds = None
        self._unique_cell_ids = None
        
        # If production cell ids don't exist we don't need to do any of the following check code
        if self._production_cell_ids is not None:
            # Production cell ids can just be reassigned if cell_ids are changed so long as there are no new cell ids
            self._production_cell_ids = pandas.Series(self._cell_ids).map(self._cid_to_pcid)
            if np.count_nonzero(self.production_cell_ids.isnull()) > 0:
                self._production_cell_ids = None
                self._pcid_to_cid = None
                self._cid_to_pcid = None
                warnings.warn("Cell ids have been changed and there are new cell ids. production_cell_ids have been set to None. If you would like new production cell ids, please run generate_production_cell_ids() again.")
            else:
                self._production_cell_ids = self.production_cell_ids.to_numpy()
        
        # If cell polygons don't exist we don't need to do any of the following check code
        if self.cell_polygons is not None:
            if self._old_cell_ids is None: # If we don't have the old cell ids we can't check if cell polygons are still valid so we just delete
                self.cell_polygons = None
                warnings.warn("Previous cell ids could not be found. Cell polygons have been removed to ensure accuracy.")
            else:
                # Make a dataframe with the counts of all cell_ids in the old and new cell ids
                #test_df = pandas.Series(self._old_cell_ids).value_counts().to_frame().join(pandas.Series(self._cell_ids).value_counts(), how='outer', lsuffix='_old', rsuffix='_new')
                test_df = pandas.DataFrame([pandas.Series(np.zeros(len(self._old_cell_ids)), index=self._old_cell_ids).groupby(level=0).indices]).T.join(pandas.DataFrame([pandas.Series(np.zeros(len(self._cell_ids)), index=self._cell_ids).groupby(level=0).indices]).T, how='outer', lsuffix='_old', rsuffix='_new')
                test_df['count_old'] = pandas.Series(self._old_cell_ids).value_counts()
                test_df['count_new'] = pandas.Series(self._cell_ids).value_counts()
                
                # Remove any polygons which have been modified
                # We specifically remove rather than set to None to distinguish cells which have not had polygons created and those for which it is not possible to create
                # differing_polygons = np.where(test_df['count_old'] != test_df['count_new'])[0]
                # for cid in test_df.index[differing_polygons]: # Loop over changed polygons
                #     self.cell_polygons.pop(cid, None)
                differing_polygons = False
                for cid, idx_old, idx_new, count_old, count_new in zip(test_df.index, test_df['0_old'], test_df['0_new'], test_df['count_old'], test_df['count_new']):
                    if count_old != count_new or np.any(idx_old != idx_new):
                        self.cell_polygons.pop(cid, None)
                        differing_polygons = True
                if differing_polygons:
                    warnings.warn("Some cells were modified, removed, or created. Cell polygons have been kept for unchanged cells but removed for modified, removed, or created cells.")
        self._old_cell_ids = None
        
        
    def unique_cell_ids(self):
        """numpy array of unique cell ids (excluding background)
        """
        if self._unique_cell_ids is None:
            unique_cell_ids = np.unique(self.cell_ids) # Pull out unique cell ids
            return np.delete(unique_cell_ids, np.where((unique_cell_ids == 0) | (unique_cell_ids == -1))) # Remove background ids
        else:
            return self._unique_cell_ids
  
    def generate_production_cell_ids(self, prefix: str|None=None, suffix: str|None=None):
        """ Generates cell ids which count up from 1 to the total cell count rather than jumping between integers.
            Production cell ids are of type string to allow for concatenating a prefix and/or suffix to the id
            If no prefix or suffix are specified, a UUID is used as a prefix
    
            Parameters
            ----------
            prefix : str | None
                String to prepend to all production cell ids
            suffix : str | None
                String to postpend to all production cell ids
    
            Sets
            ----------
            self._production_cell_ids
        """

        # Since we are assigning -1 to non-assigned transcript, we need to support negatives values
        self._cell_ids = self.cell_ids.astype(np.int64)
    
        # If neither prefix nor suffix is set, a UUID is assigned as a prefix
        if prefix is None and suffix is None:
            import uuid
            prefix = str(uuid.uuid4()) + "_"
            suffix = ''
        elif suffix is None: # If just the suffix is None, we set it to an empty string so it doesn't print out
            suffix = ''
        elif prefix is None: # If just the prefix is None, we set it to an empty string so it doesn't print out
            prefix = ''
            
        # Create dictionary to map cell ids to production ids 
        cid_to_pcid = dict(zip(self.unique_cell_ids(), np.char.mod(f'{prefix}%d{suffix}', np.arange(1, len(self.unique_cell_ids())+1))))
        cid_to_pcid[-1] = "-1"
    
        # Create dictionary to map production cell ids to cell ids 
        pcid_to_cid = dict(zip(np.char.mod(f'{prefix}%d{suffix}', np.arange(1, len(self.unique_cell_ids())+1)), self.unique_cell_ids())) 
        pcid_to_cid["-1"] = -1
    
        # Reset the current production cell ids and before we call cell_ids_changed for speed
        self._production_cell_ids = None
    
        # We set the cell ids which are 0 to -1 b/c they both mean background and we want to be consistent when we use a dictionary which goes b/w production & normal cell ids
        self._old_cell_ids = self.cell_ids.copy() # Set this so polygons stay
        self.cell_ids[self.cell_ids == 0] = -1 
        self.cell_ids_changed()
    
        self._production_cell_ids = pandas.Series(self.cell_ids).map(cid_to_pcid)
        self._pcid_to_cid = pcid_to_cid
        self._cid_to_pcid = cid_to_pcid
    
    def filter_cells(self, real_cells=None, min_spot_count=None):
        """Return a filtered spot table containing only cells matching the filter criteria.

        Parameters
        ----------
        real_cells : bool|None
            If True, include only spots that are segmented to a cell (cell IDs > 0)
        min_spot_count : int|None
            Include only spots that are segmented to cells with a minimum number of spots
        """
        assert real_cells or min_spot_count, 'One of real_cells or min_spot_count must be specified'
            
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

        cells = cells[mask]
        filtered_table = self[np.isin(self.cell_ids, cells)]
        if self.cell_polygons is not None:
            # We can copy the cell polygons over because individual cells do not change
            # We are only adding or removing cells
            filtered_table.cell_polygons = {cid: self.cell_polygons[cid] for cid in cells if cid in self.cell_polygons}

        return filtered_table

    def cell_by_gene_dense_matrix(self, dtype='uint16'):
        """Return a numpy array containing a cell-by-gene table derived from this spot table.
        Also return cell_ids and gene_ids used to construct the matrix
        """
        filtered_table = self.filter_cells(real_cells=True) # Don't want to include unassigned cells
        spot_df = filtered_table.dataframe(cols=['cell_ids', 'gene_ids']).rename({'cell_ids': 'cell', 'gene_ids': 'gene'}, axis=1)
        cell_by_gene_df = pandas.pivot_table(spot_df, columns='gene', index='cell', aggfunc=len, fill_value=0)
        
        return cell_by_gene_df.to_numpy(dtype=dtype), cell_by_gene_df.index, cell_by_gene_df.columns

    def cell_by_gene_sparse_matrix(self, dtype='uint16'):
        """Return cell-by-gene data in a scipy.sparse.csr_matrix
        Also return cell_ids and gene_ids used to construct the matrix
        """
        import scipy.sparse

        filtered_table = self.filter_cells(real_cells=True)
        
        # collect cell and gene data. we use inds b/c csr_matrix doesn't work with unordered integer names
        gene_ids = np.unique(filtered_table.gene_ids)
        cell_ids = np.unique(filtered_table.cell_ids)
        cell_id_inds = {cid:i for i,cid in enumerate(cell_ids)}
        gene_id_inds = {gid:i for i,gid in enumerate(gene_ids)}

        # count genes per cell into a dict format 
        #   {cell_index: {gene_index1: count1, ...}, ...}
        cellxgene_dict = {}
        for i in range(len(filtered_table)):
            cind = cell_id_inds[filtered_table.cell_ids[i]]
            gind = gene_id_inds[filtered_table.gene_ids[i]]
            cellrow = cellxgene_dict.setdefault(cind, {})
            cellrow.setdefault(gind, 0)
            cellrow[gind] += 1

        # convert to sparse matrix
        data = []
        rowind = []
        colind = []
        for cindex, cellrow in cellxgene_dict.items():
            for gindex, count in cellrow.items():
                data.append(count)
                rowind.append(cindex)
                colind.append(gindex)
        
        cellxgene = scipy.sparse.csr_matrix((data, (rowind, colind)), dtype=dtype)
        return cellxgene, cell_ids, gene_ids

    def cell_by_gene_anndata(self, x_format, x_dtype='uint16'):
        """Return a cell x gene table in AnnData format with cell centroids in x,y obs columns.
        """
        import anndata
        if self.production_cell_ids is None:
            raise ValueError('production_cell_ids must be set to use cell_by_gene_anndata(). See SpotTable.generate_production_cell_ids()')

        # Create the anndata.X
        if x_format == 'sparse':
            cellxgene, cell_ids, gene_ids = self.cell_by_gene_sparse_matrix(dtype=x_dtype)
        elif x_format == 'dense':
            cellxgene, cell_ids, gene_ids = self.cell_by_gene_dense_matrix(dtype=x_dtype)
        else:
            raise ValueError("x_format must be either 'dense' or 'sparse'")
        adata = anndata.AnnData(cellxgene)

        # Fill obs
        adata.obs_names = [self.convert_cell_id(cid) for cid in cell_ids]
        adata.obs = adata.obs.merge(self.cell_centroids(use_production_ids=True), how='left', left_index=True, right_index=True)
        
        # Calculate cell volumes
        if self.cell_polygons is None or len(self.cell_polygons.keys()) == 0:
            # Default to np.nan if no cell polygons
            cell_feature_df = pandas.DataFrame(index=adata.obs.index)
            cell_feature_df[['volume', 'area', 'polygon_center_x', 'polygon_center_y', 'polygon_center_z']] = np.nan
        else:
            cell_feature_df = self.get_cell_features(use_production_ids=True, disable_tqdm=True).set_index('production_cell_id')
            cell_feature_df.rename(columns={"center_x": "polygon_center_x", "center_y": "polygon_center_y", "center_z": "polygon_center_z"}, inplace=True)
        adata.obs = adata.obs.merge(cell_feature_df, how='left', left_index=True, right_index=True)
        
        adata.obs['SpotTable_cell_id'] = cell_ids

        # Fill var
        adata.var_names = self.map_gene_ids_to_names(gene_ids)
        adata.var['probe_name'] = self.map_gene_ids_to_names(gene_ids)
        adata.var['cells_with_reads'] = np.count_nonzero(cellxgene, axis=0) if x_format == 'dense' else cellxgene.getnnz(axis=0)
        segmented_total_reads = np.sum(cellxgene, axis=0).astype(int) if x_format == 'dense' else np.sum(cellxgene.A, axis=0).astype(int)
        adata.var['segmented_total_reads'] = segmented_total_reads
        unseg_table = self[self.cell_ids <= 0]
        unsegmented_total_reads = pandas.pivot_table(pandas.DataFrame({'cell': unseg_table.cell_ids, 'gene': unseg_table.gene_ids}), columns='gene', index='cell', aggfunc=len, fill_value=0).T[-1]
        unsegmented_total_reads.index = self.map_gene_ids_to_names(unsegmented_total_reads.index)
        adata.var['unsegmented_total_reads'] = unsegmented_total_reads
        
        # Fill uns
        adata.uns = {
                    'segmentation_metadata': self.seg_metadata, 
                    'cell_polygons': self.get_geojson_collection(use_production_ids=True),
                    'SIS_repo_hash': _version.get_versions()['version'],
                    }
        return adata

    def cell_bounds(self, cell_id: int | str):
        """Return ((xmin, xmax), (ymin, ymax)) for *cell_id*
        """
        if self._cell_bounds is None:
            self._cell_bounds = {}
            for cid in np.unique(self.cell_ids):
                inds = self.cell_indices(cid)
                rows = self.pos[inds]
                self._cell_bounds[cid] = ((rows[:,0].min(), rows[:,0].max()),
                                          (rows[:,1].min(), rows[:,1].max()))

        return self._cell_bounds[self.convert_cell_id(cell_id) if isinstance(cell_id, str) else cell_id]

    def cell_centroids(self, use_production_ids=False):
        """Return a Pandas DataFrame of cell centroids calculated mean of the x,y,z coordinates for each cell."""
        centroids = []
        for cid in self.unique_cell_ids():
            inds = self.cell_indices(cid)
            cell_ts_xyz = self.pos[inds]
            centroids.append(np.mean(cell_ts_xyz, axis=0))
        centroids = np.array(centroids)
        
        if centroids.shape[1] < 3: # If 2d, we still put in center_z column to keep consistent with 3d
            empty = np.zeros((len(centroids),1))
            empty[:] = np.nan
            centroids = np.append(centroids, empty, axis=1)
            
        indices = [self.convert_cell_id(cid) for cid in self.unique_cell_ids()] if use_production_ids else self.unique_cell_ids()
        return pandas.DataFrame(data=centroids, index=indices, columns=['center_x', 'center_y', 'center_z'])

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
        tri = Delaunay(cell_points_array, qhull_options="QJ")
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
        for ia, ib, ic in tri.simplices:
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
    def calculate_optimal_polygon(xy_pos, alpha_inv, alpha_inv_coeff: float=1):
        # Helper function to test if an alphashape is a polygon and contains all points
        def _test_polygon(points, polygon):
            if isinstance(polygon, shapely.geometry.polygon.Polygon):
                if not isinstance(points, shapely.geometry.MultiPoint):
                    points = shapely.geometry.MultiPoint(list(points))
                return all([polygon.intersects(shapely.geometry.Point(point)) for point in points.geoms])
            else:
                return False

        # Drop duplicate coordinates (can happen in StereoSeq data)
        xy_pos = np.unique(xy_pos, axis=0)

        if xy_pos.shape[0] > 3: # If there are <= 3 points cannot do delaunay
            putative_polygon = SegmentedSpotTable.cell_polygon(xy_pos, alpha_inv)
            # increase alpha_inv until we only have 1 polygon which contains all points
            tries = 0
            flex_alpha_inv = alpha_inv
            while tries < 200 and not _test_polygon(xy_pos, putative_polygon):
                flex_alpha_inv += .5
                tries += 1
                putative_polygon = SegmentedSpotTable.cell_polygon(xy_pos, flex_alpha_inv)

            # Final check to see if we got it in the number of tries
            if _test_polygon(xy_pos, putative_polygon):
                if alpha_inv_coeff != 1:
                    putative_polygon = SegmentedSpotTable.cell_polygon(xy_pos, alpha_inv_coeff * flex_alpha_inv)
                return putative_polygon
            else: # If not, we return the convex hull (only approximately by using massive alpha_inv)
                return SegmentedSpotTable.cell_polygon(xy_pos, 1000000)
        else:    
            return None

    def calculate_cell_polygons(self, alpha_inv=1.5, separate_z_planes=True, cells_to_run: np.ndarray|None=None, alpha_inv_coeff: float=1, disable_tqdm=False):
        if cells_to_run is None: # If you only want to calculate a subset of cells
            cells_to_run = np.unique(self.cell_ids)
            cells_to_run = np.delete(cells_to_run, np.where((cells_to_run == 0) | (cells_to_run == -1)))

        # run through all cell_ids, generate polygons and add to self.cell_polygons dict 
        # increases the alpha_inv parameter by 0.5 until a single polygon is generated
        self.cell_polygons = {}
        for cid in tqdm(cells_to_run, disable=disable_tqdm):
            inds = self.cell_indices(cid)

            if separate_z_planes:
                xyz_pos = self.pos[inds]
                for z_plane in np.unique(xyz_pos[:, 2]):
                    xy_pos = xyz_pos[xyz_pos[:, 2] == z_plane][:, :2]
                    optimal_poly = self.calculate_optimal_polygon(xy_pos, alpha_inv, alpha_inv_coeff=alpha_inv_coeff)
                    if optimal_poly: # Only record a polygon if a plane had a polygon (i.e. don't store None)
                        self.cell_polygons.setdefault(cid, {})[z_plane] = optimal_poly
                self.cell_polygons.setdefault(cid, None) # If none of the z-planes had a polygon, set to None
            else:
                xy_pos = self.pos[inds][:, :2]
                self.cell_polygons[cid] = self.calculate_optimal_polygon(xy_pos, alpha_inv, alpha_inv_coeff=alpha_inv_coeff)


    @staticmethod
    def calculate_cell_features(cell_polygon, z_plane_thickness=1.5):
        if isinstance(cell_polygon, dict): # If we have separate polygons for each z-plane
            area = 0
            weighted_x, weighted_y, weighted_z = 0, 0, 0
            for z, polygon in cell_polygon.items():
                if polygon:
                    area += polygon.area
                    
                    weighted_x += polygon.centroid.coords[0][0] * polygon.area
                    weighted_y += polygon.centroid.coords[0][1] * polygon.area
                    weighted_z += z * polygon.area
                    
            volume = area * z_plane_thickness
            center_x, center_y, center_z = weighted_x / area, weighted_y / area, weighted_z / area # we define centroid as a weighted average (by area) of centroids of polygons
            
            return {"volume": volume, "center_x": center_x, "center_y": center_y, "center_z": center_z}
        else: # If we have one polygon for each cell (either one representing all z-planes or just one z-plane)
            return {"area": cell_polygon.area, "center_x": cell_polygon.centroid.coords[0][0], "center_y": cell_polygon.centroid.coords[0][1]}
    
    
    def get_cell_features(self, z_plane_thickness=1.5, use_production_ids: bool=False, use_both_ids: bool=False, disable_tqdm=False):
        # run through self.polys and calculate features
        if self.cell_polygons is None or len(self.cell_polygons.keys()) == 0:
            raise ValueError("Cell polygons must be set before calculating cell features. See calculate_cell_polygons() or load_cell_polygons()")

        cell_features = []
        for cid in tqdm(self.cell_polygons, disable=disable_tqdm):
            feature_info = {"cell_id":cid, "volume": np.nan, "area": np.nan, "center_x": np.nan, "center_y": np.nan, "center_z": np.nan}
            if use_both_ids:
                feature_info.update({"production_cell_id": self.convert_cell_id(cid)})
            if use_production_ids:
                feature_info.update({"production_cell_id": self.convert_cell_id(feature_info.pop("cell_id"))})
            
            if self.cell_polygons[cid]: # Sometimes polygons can be None
                feature_info.update(self.calculate_cell_features(self.cell_polygons[cid], z_plane_thickness=z_plane_thickness))

            cell_features.append(feature_info)
        
        # Depending if we have separate polygons for each z-plane we want to set either the volume or area to NaN
        df = pandas.DataFrame.from_records(cell_features)
        
        return df


    def get_geojson_collection(self, use_production_ids=False):
        """
        Create a geojson feature/geometry collection from the cell polygons
        """
        if self.cell_polygons is None:
            return None
        elif dict in set(type(k) for k in self.cell_polygons.values()): # if cell polygons are separated by z-plane use feature collection which stores z-plane info
            all_polygons = []
            for cid in self.cell_polygons:
                if self.cell_polygons[cid]:
                    for z_plane, polygon in self.cell_polygons[cid].items():
                        # Each z-plane is a separate feature
                        all_polygons.append(geojson.Feature(geometry=polygon, id=self.convert_cell_id(cid) if use_production_ids else str(cid), z_plane=str(z_plane)))
                else:
                    all_polygons.append(geojson.Feature(geometry=None, id=self.convert_cell_id(cid) if use_production_ids else str(cid), z_plane=None))

            return geojson.FeatureCollection(all_polygons)
        else: # If cell polygons are not separated by z-plane, use previous implementation
            geojsonROIs = []
            for poly_key in self.cell_polygons.keys():
                if self.cell_polygons[poly_key]:
                    if len(self.cell_polygons[poly_key].exterior.coords) > 2:
                        geojsonROIs.append(util.poly_to_geojson(self.cell_polygons[poly_key]))
                else:
                    geojsonROIs.append(None)
            
            return geojson.GeometryCollection(geojsonROIs)
    

    def save_cell_polygons(self, save_path: Path|str, use_production_ids=False):
        """
        save a geojson geometry collection from the cell polygons
        """
        # Handle input errors
        if isinstance(save_path, Path) or isinstance(save_path, str):
            extension = Path(save_path).suffix
            if extension != '.pkl' and extension != '.geojson':
                raise ValueError('Invalid path extension. Please use .pkl or .geojson')
        else:
            raise ValueError('Invalid path type. Please use pathlib.Path or str')

        # Raise error if no data
        if self.cell_polygons is None or len(self.cell_polygons) == 0:
            raise ValueError('No cell polygon data, cannot save file')
        
        if extension == '.geojson':
            with open(save_path, "w") as f:
                geojson.dump(self.get_geojson_collection(use_production_ids=use_production_ids), f)
        else:
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(self.cell_polygons, f)

    
    def load_cell_polygons(self, load_path: Path|str, cell_ids: Path|str|list|np.ndarray|None=None, reset_cache=True, disable_tqdm=False):
        """
        load cell polygons from a geojson feature collection file

        Parameters
        ----------
        load_path : Path|str
            Path to the file containing the cell polygons. Can be either '.geojson' or '.pkl'
        cell_ids [OPTIONAL]: Path|str|list|np.ndarray|None
            Path/str to .npy file containing cell ids corresponding to cell polygons for GeometryCollection
            list/np.ndarray containing cell ids corresponding to cell polygons for GeometryCollection
            Used to assign polygons to cell ids for GeometryCollection b/c it does not store IDs. 
            This is important if you want to load in a bunch of 2D polygons files w/o reseting the cache as we have no way to determine proper ID
            If not provided and using GeometryCollection, it is assumed the cell polygons are in order of the sorted cell ids
            GeometryCollections may contain None for uncalculated polygons or may exclude them entirely
        """
        # Handle input errors
        if isinstance(load_path, Path) or isinstance(load_path, str):
            extension = Path(load_path).suffix
            if extension != '.pkl' and extension != '.geojson':
                raise ValueError('Invalid path extension. Can only load .pkl or .geojson')
        else:
            raise ValueError('Invalid path type. Please use pathlib.Path or str')

        if reset_cache or self.cell_polygons is None:
            self.cell_polygons = {}

        if extension == '.geojson': 
            import json
            from shapely.geometry.polygon import Polygon

            with open(load_path, "r") as f:
                polygon_json = json.load(f)

            unique_cells = np.unique(self.cell_ids)
            unique_cells = np.delete(unique_cells, np.where((unique_cells == 0) | (unique_cells == -1)))
            cell_id_type = type(unique_cells[0])
            z_plane_type = None if self.pos.shape[1] < 3 else type(self.pos[0, 2])

            if polygon_json['type'] == 'FeatureCollection':
                for feature in tqdm(polygon_json['features'], disable=disable_tqdm):
                    cid = cell_id_type(feature['id'])
                    if cid in unique_cells:
                        # Make sure it is a polygon or None otherwise we don't read it
                        if feature['geometry'] and feature['geometry']['type'] == 'Polygon': 
                            z_plane = z_plane_type(feature['z_plane'])
                            polygon = Polygon(feature['geometry']['coordinates'][0])
                            self.cell_polygons.setdefault(cid, {})[z_plane] = polygon
                        elif not feature['geometry']:
                            self.cell_polygons[cid] = feature['geometry']
            elif polygon_json['type'] == 'GeometryCollection':
                from pathlib import PurePath
                if cell_ids and (isinstance(cell_ids, PurePath) or isinstance(cell_ids, str)):
                    cell_ids = list(np.load(cell_ids))
                elif cell_ids and isinstance(cell_ids, np.ndarray):
                    cell_ids = list(cell_ids)

                if cell_ids:
                    for geometry in tqdm(polygon_json['geometries'], disable=disable_tqdm):
                        if geometry and geometry['type'] == 'Polygon':
                            polygon = Polygon(geometry['coordinates'][0])
                            self.cell_polygons[cell_ids.pop(0)] = polygon
                        elif not geometry:
                            self.cell_polygons[cell_ids.pop(0)] = geometry
                else: # If we don't have the cell IDs we will have to infer
                    if len(unique_cells) < len(polygon_json['geometries']):
                        raise ValueError("Number of cells in input file exceeds SpotTable")
                        
                    # This method ensure compatibility with both JSONs which store None and those which dont
                    valid_cells = [cid for cid in unique_cells if len(self.cell_indices(cid)) > 3]
                    invalid_cells = [cid for cid in unique_cells if len(self.cell_indices(cid)) <= 3]
                    
                    for geometry in tqdm(polygon_json['geometries'], disable=disable_tqdm):
                        if geometry and geometry['type'] == 'Polygon':
                            polygon = Polygon(geometry['coordinates'][0])
                            self.cell_polygons[valid_cells.pop(0)] = polygon
                        elif not geometry:
                            self.cell_polygons[invalid_cells.pop(0)] = geometry
            else:
                raise ValueError('geojson type must be FeatureCollection or GeometryCollection')
        else:
            import pickle
            with open(load_path, "rb") as f:
                self.cell_polygons.update(pickle.load(f))

    
    def cell_indices(self, cell_ids: int | str | np.ndarray):
        """Return indices giving table location of all spots with *cell_ids*
        """
        if self._cell_index is None:
            ind = {}
            for i, cid in enumerate(self.cell_ids):
                ind.setdefault(cid, []).append(i)
            self._cell_index = ind

        if isinstance(cell_ids, (int, np.integer)):
            return self._cell_index[cell_ids]
        elif isinstance(cell_ids, str):
            return self._cell_index[self.convert_cell_id(cell_ids)]
        else:
            if len(cell_ids) == 0:
                return np.array([], dtype=int)

            if isinstance(cell_ids[0], str):
                cell_ids = np.vectorize(self._pcid_to_cid.get)(cell_ids)

            return np.concatenate([self._cell_index[cid] for cid in cell_ids])

    def cells_inside_region(self, xlim: tuple, ylim: tuple):
        """Return IDs of cells that are entirely inside xlim, ylim.
        """
        cell_ids = []
        for cid in np.unique(self.cell_ids):
            (x0, x1), (y0, y1) = self.cell_bounds(cid)
            if x0 > xlim[0] and x1 < xlim[1] and y0 > ylim[0] and y1 < ylim[1]:
                cell_ids.append(cid)
        return cell_ids

    def cell_mask(self, cell_ids: int | str | np.ndarray):
        """Return a mask selecting spots that belong to cells in *cell_ids*
        """
        mask = np.zeros(len(self), dtype=bool)

        if np.issubdtype(type(cell_ids), np.integer):
            mask[self.cell_indices(cell_ids)] = True
        elif isinstance(cell_ids, str):
            mask[self.cell_indices(self.convert_cell_id(cell_ids))] = True
        else:
            if isinstance(cell_ids[0], str):
                cell_ids = np.vectorize(self._pcid_to_cid.get)(cell_ids)

            for cid in cell_ids:
                mask[self.cell_indices(cid)] = True

        return mask

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

    def merge_cells(self, other, padding=5, union_threshold=0.5):
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

        # At this point the primary merge is done the rest is solving conflicts
        new_state = self[other.parent_inds]
        affected_cell_ids = set(original_state.cell_ids[tile_inds]) - set([-1, 0]) # cells in original state that were affected by merge
        new_cell_ids = set(new_state.cell_ids) - set([-1, 0]) # all cells present in new merge area

        # set of cells that were partially replaced by the merge
        # (cells that were affected by the merge, but not completely replaced by the merge)
        partial_merge_cells = new_cell_ids & affected_cell_ids

        conflicts = []
        for cell_id in partial_merge_cells:
            old_inds = original_state.cell_indices(cell_id)
            new_inds = new_state.cell_indices(cell_id)
        
            overlapped_cells = set(new_state.cell_ids[old_inds]) - set([-1, 0])
            overlapped_cell_inds = {cid:new_state.cell_indices(cid) for cid in overlapped_cells}
        
            original_cell_size = len(old_inds)
            reduced_cell_size = len(new_inds)
            overlapped_cell_sizes = {cid:len(inds) for cid, inds in overlapped_cell_inds.items()}
            overlapped_cell_parent_inds = {cid:new_state.map_indices_to_parent(inds) for cid, inds in overlapped_cell_inds.items()}
            overlaps = {}
            overlap_pct = {}
            to_combine = []
            for overlap_cell, size in overlapped_cell_sizes.items():
                if overlap_cell == cell_id: continue
            
                overlap_size = np.count_nonzero(np.in1d(overlapped_cell_inds[overlap_cell], old_inds))
                if overlap_size / min(original_cell_size, size) > union_threshold: # if the overlap transcripts represents >threshold of the transcripts of the smaller cell i.e. the smaller cell is mostly 'absorbed'
                    to_combine.append(overlap_cell)
                overlaps[overlap_cell] = overlap_size
                overlap_pct[overlap_cell] = overlap_size / min(original_cell_size, size)

            new_indices = new_state.map_indices_to_parent(new_state.cell_indices([overlap_cell for overlap_cell in to_combine]+ [cell_id]))
            self.cell_ids[new_indices] = cell_id
            self.cell_ids_changed()
            
            # Keep track of the conflict resolution process
            conflicts.append({
                'original_cell_id': cell_id,
                'original_size': original_cell_size,
                'size_after_merge': reduced_cell_size,
                'original_indices': new_state.map_indices_to_parent(old_inds),
                'overlapped_cells': overlapped_cells,
                'overlapped_cell_sizes': overlapped_cell_sizes,
                'overlapped_cell_indices': overlapped_cell_parent_inds,
                'merged_cells': to_combine,
                'merged_cell_overlap_sizes': overlaps,
                'merged_cell_overlap_ratios': overlap_pct,
                'new_indices': new_indices
            })

        return conflicts

    def set_cell_ids_from_tiles(self, tiles, padding=5):
        """Overwrite all cell IDs by merging from *tiles*, which may be a list of SegmentedSpotTable
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

    @classmethod
    def load_baysor(cls, file_name: str, **kwds):
        """Return a new SegmentedSpotTable loaded from a baysor result file.
        """
        from .segmentation import load_baysor_result
        result = load_baysor_result(file_name, **kwds)
        pos = result[['x', 'y', 'z']].view(dtype=result.dtypes.fields['x'][0]).reshape(len(result), 3)
        spot_table = SpotTable(pos=pos, gene_ids=result['gene'])
        return cls(spot_table, cell_ids=result['cell'], seg_metadata={'seg_method': 'Baysor'})

    @staticmethod
    def load_merscope_cell_ids(csv_file: str, max_rows: int|None=None):
        """Load the original segmentation for a MERSCOPE dataset.

        Parameters
        ----------
        csv_file : str
            Path to the detected transcripts file.
        max_rows : int, optional
            Maximum number of rows to load from the CSV file.

        Returns
        -------
        numpy.ndarray
        """
        print('Loading MERSCOPE cell ids...')
        with open(csv_file, 'r') as f:
            cols_in_file = f.readline().split(',')
        cols_in_file = [col.strip() for col in cols_in_file]
        cell_col_ind = cols_in_file.index('cell_id')
        cell_ids = np.loadtxt(csv_file, skiprows=1, usecols=cell_col_ind, delimiter=',', dtype='int64', max_rows=max_rows)
        return cell_ids

    @classmethod
    def load_merscope(cls, csv_file: str, cache_file: str|None, image_path: str|None=None, max_rows: int|None=None):
        """Load MERSCOPE data from a detected transcripts CSV file, including
        the original segmentation. If you are resegmenting the data, prefer
        SpotTable.load_merscope.

        Note: If cache_file is set, only the raw spot table is cached, not the 
        cell_ids. This is for consistency with SpotTable.load_merscope.

        Parameters
        ----------
        csv_file : str
            Path to the detected transcripts file.
        cache_file : str, optional
            Path to the detected transcripts cache file, which is an npz file 
            representing the raw SpotTable (without cell_ids). If passed, will
            create a cache file if one does not already exists.
        image_path : str, optional
            Path to directory containing a MERSCOPE image stack.
        max_rows : int, optional
            Maximum number of rows to load from the CSV file.

        Returns
        -------
        sis.spot_table.SegmentedSpotTable
        """
        raw_spot_table = SpotTable.load_merscope(csv_file=csv_file, cache_file=cache_file, image_path=image_path, max_rows=max_rows)
        cell_ids = SegmentedSpotTable.load_merscope_cell_ids(csv_file, max_rows=max_rows)

        return cls(spot_table=raw_spot_table, cell_ids=cell_ids, seg_metadata={'seg_method': 'MERSCOPE'})


    @classmethod
    def load_xenium(cls, transcript_file: str, cache_file: str|None, image_path: str|None=None, max_rows: int|None=None, z_depth: float=3.0):
        """Load Xenium data from a detected transcripts CSV file, including
        the original segmentation. If you are resegmenting the data, prefer
        SpotTable.load_xenium.

        Note: If cache_file is set, only the raw spot table is cached, not the 
        cell_ids. This is for consistency with SpotTable.load_xenium.

        Parameters
        ----------
        transcript_file : str
            Path to the detected transcripts file.
        cache_file : str, optional
            Path to the detected transcripts cache file, which is an npz file 
            representing the raw SpotTable (without cell_ids). If passed, will
            create a cache file if one does not already exists.
        image_path : str, optional
            Path to directory containing a Xenium image stack.
        max_rows : int, optional
            Maximum number of rows to load from the CSV file.
        z_depth : float, optional
            Depth (in um) of a imaging layer i.e. z-plane
            Used to bin z-positiions into discrete planes

        Returns
        -------
        sis.spot_table.SegmentedSpotTable
        """
        raw_spot_table = SpotTable.load_xenium(transcript_file=transcript_file, cache_file=cache_file, image_path=image_path, max_rows=max_rows, z_depth=z_depth)

        if str(transcript_file).endswith('.csv') or str(transcript_file).endswith('.csv.gz'):
            cell_ids = pandas.read_csv(transcript_file, nrows=max_rows, usecols=['cell_id']).values

        elif str(transcript_file).endswith('.parquet'):
            if max_rows:
                raise ValueError('max_rows is not supported for parquet files as pandas does not allow partial reading')

            cell_ids = pandas.read_parquet(transcript_file, columns=['cell_id']).values

        cell_ids = np.squeeze(cell_ids)

        if not np.issubdtype(cell_ids.dtype, np.integer):
            # Xenium cell_id column is string, but SegmentedSpotTable needs ints, so convert
            unique_ids = np.unique(cell_ids)
            bg_id = 'UNASSIGNED'
            assert bg_id in unique_ids  # if this changes with Xenium version, may need more options
            if unique_ids[0] != bg_id:
                # put background cell id first so it will be paired with 0
                temp = set(unique_ids).difference([bg_id])
                cell_ids = [bg_id] + list(temp)

            cid_to_ints_mapping = dict(zip(unique_ids, np.arange(0, len(unique_ids))))
            cell_ids = [cid_to_ints_mapping[cid] for cid in cell_ids]
            cell_ids = np.array(cell_ids)

        return cls(spot_table=raw_spot_table, cell_ids=cell_ids, seg_metadata={'seg_method': 'Xenium'})


    @classmethod
    def load_stereoseq(cls, gem_file: str|None=None, cache_file: str|None=None, gem_cols: dict|tuple=(('gene', 0), ('x', 1), ('y', 2), ('MIDcounts', 3)), 
                       cell_col: int|None=None, skiprows: int|None=1,  max_rows: int|None=None, image_file: str|None=None, image_channel: str|None=None):
        """
        Load StereoSeq data from gem file. This can be slow so optionally cache the result to a .npz file.
        1/19/2023: New StereoSeq data has cell_ids, add optional cell_cols to add to SpotTable. Also has a flag
        for whether the spot was in the main cell or the extended cell (in_cell)
        1/3/2024: new cellbin.gem doesn't have `in_cell`, let's not worry about it and just specify the column 
        that the cell ID is in
        """
        raise NotImplementedError('Loading native cell ids for StereoSeq is not supported. Use SpotTable.load_stereoseq to load detected transcripts.')

    @classmethod
    def load_npz(cls, npz_file: str, images: ImageBase|list[ImageBase]|None=None, allow_pickle: bool=False):
        """Load from an NPZ file.

        Parameters
        ----------
        npz_file : str
            Path to the npz file.
        images : ImageBase or list of ImageBase, optional
            Image(s) to attach to the SpotTable. Must be loaded separately
            since these cannot be stored in the NPZ file.
        allow_pickle : bool, optional
            Whether to allow loading pickled object arrays stored in npy files.
            Must be enabled to load dictionaries and cell polygons.

        Returns
        -------
        sis.spot_table.SegmentedSpotTable
        """
        fields = np.load(npz_file, allow_pickle=allow_pickle)
        spot_table = SpotTable(
                pos=fields['pos'],
                gene_ids=fields['gene_ids'],
                gene_id_to_name=fields['gene_id_to_name'],
                images=images
                )
        
        # handle underscores and object arrays
        sst_fields = {
            'cell_ids': fields['cell_ids'],
            'seg_metadata': fields['seg_metadata'].item(),
            'production_cell_ids': fields['production_cell_ids'].item() if np.any(fields['production_cell_ids']) is None else fields['production_cell_ids'],
            'pcid_to_cid': fields['_pcid_to_cid'].item(),
            'cid_to_pcid': fields['_cid_to_pcid'].item(),
            'cell_polygons': fields['cell_polygons'].item()
        }

        return cls(spot_table=spot_table, **sst_fields)

    def save_npz(self, npz_file: str):
        """Save to an NPZ file.

        Parameters
        ----------
        npz_file : str
            Output path for the npz file.
        """
        fields = {
            'pos': self.pos,
            'gene_ids': self.gene_ids, 
            'gene_id_to_name': self.gene_id_to_name,
            'cell_ids': self.cell_ids,
        }

        kwds = ['seg_metadata', 'production_cell_ids', '_pcid_to_cid', '_cid_to_pcid', 'cell_polygons']
        fields.update({kwd: getattr(self, kwd) for kwd in kwds})

        np.savez_compressed(npz_file, **fields) 

    def dataframe(self, cols=['x', 'y', 'z', 'gene_ids', 'cell_ids']):
        """Return a dataframe containing the specified columns.

        By default, columns are x, y, z, gene_ids, and cell_ids.
        Also available: gene_names
        """
        if 'cell_ids' in cols:
            cols.remove('cell_ids')
            df = self.spot_table.dataframe(cols)
            df['cell_ids'] = self.cell_ids
        else:
            df = self.spot_table.dataframe(cols)
        return df 

    def cell_scatter_plot(self, *args, **kwds):
        """Scatter plot of spots colored by cell ID
        """
        kwds['color'] = 'cell_ids'
        kwds['palette'] = self.cell_palette(self.cell_ids)
        return self.scatter_plot(*args, **kwds)

    def scatter_plot(self, ax=None, x='x', y='y', color='gene_ids', alpha=0.2, size=1.5, z_slice=None, palette=None):
        import seaborn
        import matplotlib.pyplot as plt
        if color in ['cell', 'cell_ids']:
            palette = self.cell_palette(self.cell_ids)

        if ax is None:
            fig, ax = plt.subplots()
        
        if z_slice is not None:
            zvals = np.unique(self.z)
            zval = zvals[int(z_slice * (len(zvals)-1))]
            mask = self.z == zval
            self = self[mask]

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

    def copy(self, deep:bool=False, **kwds):
        """Return a copy of self, optionally with some attributes replaced.
        Currently doesn't support replacing attributes of SpotTable (I'm not
        sure what the use case for this would be) but this could be added if
        needed.
        """
        spot_table = self.spot_table.copy(deep=deep)
        init_kwargs = {}
        init_kwargs.update(kwds)
        for name in ['cell_ids', 'production_cell_ids', '_pcid_to_cid', '_cid_to_pcid', 'cell_polygons', 'seg_metadata']:
            if name not in init_kwargs:
                val = getattr(self, name)
                if deep:
                    val = None if val is None else val.copy(),
                if name.startswith('_'): # This is to handle _pcid_to_cid and _cid_to_pcid as arguments
                    name = name[1:]
                init_kwargs[name] = val
            
        return SegmentedSpotTable(spot_table=spot_table, **init_kwargs)

    def get_subregion(self, xlim: tuple, ylim: tuple, incl_end: bool=False):
        """Return a SegmentedSpotTable including the subset of this table inside the region xlim, ylim
        """
        subtable = self.spot_table.get_subregion(xlim, ylim, incl_end)
        seg_subtable = self[subtable.parent_inds]
        seg_subtable.spot_table.images = [img.get_subregion(subtable.parent_region, incl_end=incl_end) for img in subtable.images]

        return seg_subtable
