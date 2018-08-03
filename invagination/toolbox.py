"""
Some utils function
"""
import os
import warnings
import math
import numpy as np
import pandas as pd


from tyssue import Sheet, config
from tyssue.io import hdf5


def open_sheet(dirname, t):
    """Open hdf5 file

    Open HDF5 file correspond to t time from dirname directory.

    Parameters
    ----------
    directory : str
        complete directory path
    t : int
        time step
    """
    file_name = 'invagination_{:04d}.hf5'.format(t)
    dsets = hdf5.load_datasets(os.path.join(dirname, file_name),
                               data_names=['vert', 'edge', 'face', 'cell'])

    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet('ellipse', dsets, specs)
    return sheet


def define_depth(directory, t, coord=['x', 'y'],
                 xmin=-20, xmax=20, ymin=-2, ymax=2):
    """Define depth for t time from directory.
    Parameters
    ----------
    directory : str
        complete directory path
    t : int
        time step
    coord : list

    xmin: int
    xmax: int
    ymin: int
    ymax: int
        Position of the area where the measure have to be done.
    """

    sheet = open_sheet(directory, t)
    try:
        sheet_mesoderm = sheet.extract('is_mesoderm')
    except:
        sheet_mesoderm = sheet.extract('is_fold_patch')
    subset_mesoderm_face_df = sheet_mesoderm.face_df[
        (sheet_mesoderm.face_df[coord[0]] > xmin) &
        (sheet_mesoderm.face_df[coord[0]] < xmax) &
        (sheet_mesoderm.face_df[coord[1]] > ymin) &
        (sheet_mesoderm.face_df[coord[1]] < ymax)]

    depth = np.mean(subset_mesoderm_face_df['rho'])

    return depth


def define_time_max_depth(directory, nb_t=200):
    """Define time and depth where the depths
    is maximal in a directory
    """
    depth_0 = define_depth(directory, 0, ['z', 'x'])

    depths = []
    for t in range(0, nb_t):
        try:
            depths.append(depth_0 - define_depth(directory, t, ['z', 'x']))
        except Exception:
            depths.append(0)
            warnings.warn(
                'An error occured at time %i for directory %s', t, directory)

    time = depths.index(max(depths))
    return time, max(depths)


def define_time_depth_compare_to_vivo(directory, nb_t=200, in_vivo_depth=4.5):
    """Define the time when the depth is the closest from in vivo depth.
    """
    depth_0 = define_depth(directory, 0, ['z', 'x'])

    depths = []
    for t in range(0, nb_t):
        depths.append(depth_0 - define_depth(directory, t, ['z', 'x']))

    depths = [0 if math.isnan(x) else x for x in depths]

    val_depth = min(depths, key=lambda x: abs(x - in_vivo_depth))
    time = depths.index(val_depth)
    return time, val_depth


def force_ratio(sheet, critical_area=5):
    """Returns the ration of contration to apicobasal tension for
    constricting cells.
    """
    pulling_faces = sheet.face_df[
        (sheet.face_df.area < critical_area) &
        (sheet.face_df.is_mesoderm)].index
    contraction = sheet.face_df.loc[
        pulling_faces].eval('perimeter * contractility')
    tension = sheet.vert_df.radial_tension
    ratio = tension.sum() / contraction.sum()
    return ratio


def face_centered_patch(sheet, face, neighbour_order):

    faces = sheet.get_neighborhood(face, order=neighbour_order)['face']
    edges = sheet.edge_df[sheet.edge_df['face'].isin(faces)]

    vertices = sheet.vert_df.loc[set(edges['srce'])]
    pos = vertices[sheet.coords].values - \
        vertices[sheet.coords].mean(axis=0).values[None, :]
    u, v, rotation = np.linalg.svd(pos, full_matrices=False)
    rot_pos = pd.DataFrame(np.dot(pos, rotation.T),
                           index=vertices.index,
                           columns=sheet.coords)

    patch_dset = {'vert': rot_pos,
                  'face': sheet.face_df.loc[faces].copy(),
                  'edge': edges.copy()}

    patch = Sheet('patch', patch_dset, sheet.specs)
    patch.reset_index()
    return patch

#patch = face_centered_patch(sheet, 190, 2)
#fig, ax = sheet_view(patch, mode='quick', coords=['x', 'y'])
#    patch = face_centered_patch(sheet, 53, 2)
#    fig, ax = quick_edge_draw(patch, ['x', 'y'],
#                              alpha=0.7)
