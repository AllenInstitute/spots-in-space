import numpy as np


def reduce_expression(data, umap_args):
    import umap
    from sklearn.preprocessing import StandardScaler

    default_umap_args = {'n_neighbors': 3, 'min_dist': 0.4, 'n_components': 3}
    default_umap_args.update(umap_args)

    flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    # randomize order (because umap has some order-dependent effects)
    order = np.arange(flat_data.shape[0])
    np.random.shuffle(order)
    flat_data = flat_data[order]

    # remove rows with no transcripts
    mask = flat_data.sum(axis=1) > 0
    masked_data = flat_data[mask]

    # scale in prep for umap
    scaler = StandardScaler()
    scaled = scaler.fit_transform(masked_data)

    # reduce down to 3D
    reducer = umap.UMAP(**default_umap_args)
    reduced = reducer.fit_transform(scaled)

    # re-insert rows with no transcripts (all 0)
    final = np.zeros((len(flat_data), reduced.shape[1]), dtype=reduced.dtype)
    final[mask] = reduced

    # un-shuffle order
    reverse_order = np.argsort(order)
    final = final[reverse_order]

    # return reshaped to original image
    return final.reshape(data.shape[0], data.shape[1], final.shape[-1])


def map_to_ubyte(data):
    mn, mx = data.min(), data.max()
    return np.clip((data - mn) * 255 / (mx - mn), 0, 255).astype('ubyte')


def rainbow_wheel(points, center=None, radius=None, center_color=None):
    """Given an Nx2 array of point locations, return an Nx3 array of RGB
    colors derived from a rainbow color wheel centered over the mean point location.
    """
    import matplotlib.pyplot as plt
    import scipy.interpolate
    flat = points.reshape(np.product(points.shape[:-1]), points.shape[-1])
    if center is None:
        center = flat.mean(axis=0)
    if radius is None:
        radius = 4 * flat.std(axis=0)
    f = np.linspace(0, 1, 10)[:-1]
    theta = f * 2 * np.pi
    x = np.vstack([radius[0] * np.cos(theta) + center[0], radius[1] * np.sin(theta) + center[1]]).T
    c = plt.cm.gist_rainbow(f)[:, :3]

    if center_color is not None:
        x = np.concatenate([x, center[None, :]], axis=0)
        c = np.concatenate([c, np.array(center_color)[None, :]], axis=0)

    color = scipy.interpolate.griddata(x, c, flat[:, :2], fill_value=0)
    return color.reshape(points.shape[:-1] + (3,))


def show_float_rgb(data, extent, ax):
    """Show a color image given a WxHx3 array of floats.
    Each channel is normalized independently.
    """
    rgb = np.empty(data.shape[:2] + (3,), dtype='ubyte')
    for i in (0, 1, 2):
        rgb[..., i] = map_to_ubyte(data[..., i])

    return ax.imshow(rgb, extent=extent, aspect='equal', origin='lower')


def log_plus_1(x):
    return np.log(x + 1)


def poly_to_geojson(polygon):
    """
    turns a single shapely Polygon into a geojson polygon
    Args:
        polygon shapely.Polygon
    Returns:
        geojson polygon
    """
    import geojson
    poly_array = np.array(polygon.exterior.coords)

    return geojson.Polygon([[(poly_array[i,0], poly_array[i,1]) for i in range(poly_array.shape[0])]])

def load_config(configfile=None):
    import yaml
    import os

    if configfile is None:
        configfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spatial_config.yml'))
        
    if os.path.isfile(configfile):
        if hasattr(yaml, 'FullLoader'):
            # pyyaml new API
            config = yaml.load(open(configfile, 'rb'), Loader=yaml.FullLoader)
        else:
            # pyyaml old API
            config = yaml.load(open(configfile, 'rb'))

    else:
        config = {}
    return config