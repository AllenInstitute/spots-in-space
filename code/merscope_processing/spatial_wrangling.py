# +
import pandas as pd
import numpy as np

import pathlib

from scipy.spatial import Delaunay

from shapely.geometry import MultiLineString, box, Polygon, Point, MultiPolygon
from shapely.ops import unary_union, polygonize

import shapely
import time

from matplotlib import pyplot as plt

import anndata as ad



def spots_in_polygon(spot_dataframe, spot_coordinates, polygon,region_name: str="polygon", checkBoundingBox = False):
    """
    modifies input dataframe to add column indicating if each spot is in the input polygon.  
    currently only checks in 2D
    uses shapely's contains() method

    Args:
        spot_dataframe is a pandas dataframe
        spot_coordinates are the columns of spot_dataframe to be used for the s
        polygon is a  shapely.geometry.Polygon
        region_name: string that will be used in the new column name
        checkBoundingBox if True, use bounding box coordinates to avoid making a containsPoint call.

    Returns:
        spots_dataframe
    """
    
    in_region_string = "is_in_"+region_name
    if in_region_string not in spot_dataframe.columns:
        print(in_region_string+" not in columns... adding to spot_dataframe")
        spot_dataframe[in_region_string] = False

    if checkBoundingBox:

        polySpots =[]
        bb_bounds = polygon.bounds
        conditionA = np.logical_or(spot_dataframe[spot_coordinates[0]].values< bb_bounds[0] ,
                         spot_dataframe[spot_coordinates[0]].values> bb_bounds[2])
        conditionB = np.logical_or(spot_dataframe[spot_coordinates[1]].values< bb_bounds[1] ,
                         spot_dataframe[spot_coordinates[1]].values> bb_bounds[3]) 

        conditions = np.logical_not( np.logical_or(conditionA, conditionB))
        # this is the subset of spots within the xy bounding box of the polygon:


        okSpots = spot_dataframe.loc[conditions,spot_coordinates].values
        
    else:
        
        okSpots = spot_dataframe.loc[:,spot_coordinates].values

        
    spotList = []

    for i in range(okSpots.shape[0]):
        spotList.append( Point(okSpots[i,0:2]))
            
    
    is_in_roi = [polygon.contains(spot) for spot in spotList] 

    
    if checkBoundingBox:
        spot_dataframe.loc[conditions,in_region_string] = np.logical_or(is_in_roi,spot_dataframe.loc[conditions,in_region_string])
    else:
        spot_dataframe.loc[:,in_region_string] = np.logical_or(is_in_roi,spot_dataframe.loc[:,in_region_string])
    
    return spot_dataframe


def getAlpha2D(  alphaInv, points=None):
    """
    get 2D alpha shape from a set of points, modified slightly from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/


    args:
    points   should be numpy array of points with expected columns [x,y].
    alphaInv parameter that sets the radius filter on the Delaunay triangulation.  traditionally alpha is defined as 1/radius, and here the input is inverted for slightly more intuitive use
    """


    if points.shape[0]< 4:
        return points, None

    tri = Delaunay(points)
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
        edge_points.append(points[ [i, j] ])


    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

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
        if circum_r < alphaInv:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    tp = unary_union(triangles)
    polygons = []
    if isinstance(tp,shapely.geometry.MultiPolygon)  :
        for p in tp.geoms:
            polygons.append(Polygon( np.array(list(p.exterior.coords))))
            
        
    elif isinstance(tp,shapely.geometry.GeometryCollection):
        
            print("encountered GeometryCollection inside MultiPolygon")
        
    else:
        polygons.append( Polygon(np.array(list(tp.exterior.coords))))

    return polygons


def spatial_subset(h5ad_path: pathlib.Path, 
                   spot_csv_path: pathlib.Path,
                   clean_region_string: str,
                   region_buffer = 50,
                  alpha_radius = 50,
                  output_path = None):
    
    spot_table =  pd.read_csv(spot_csv_path)


    
    # single section data:
    section_cells = ad.read_h5ad(h5ad_path)

    region_subset = section_cells.obs["clean_region_label"]==clean_region_string


    region_polygon = getAlpha2D(alpha_radius,points = section_cells[region_subset].obsm["spatial"])


    print("found "+str(len(region_polygon))+" polygons")
    for p in region_polygon:
        pbigger = p.buffer(region_buffer)

        t0 = time.time()
        print("starting spot assignment...")
        spots_in_polygon(spot_table, ["global_x","global_y"], pbigger,region_name=clean_region_string, checkBoundingBox = True)

        print("spot assignment time: "+str(time.time()-t0))
    if output_path:
        spot_table.to_csv(output_path)
        return None
    else:
        return spot_table
# -


