import json
import numpy as np
import pandas as pd

from tyssue import Sheet
from tyssue.io import hdf5
from tyssue.utils.utils import _to_3d, to_nd
from tyssue.dynamics.sheet_gradients import height_grad
from tyssue.dynamics import units, effectors, model_factory


class VitellineElasticity(effectors.AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = 'vitelline_K'
    label = 'Vitelline elasticity'
    element = 'vert'
    specs = {
        'vert': {
            'vitelline_K',
            'is_active',
            'delta_rho'}
    }  # distance to the vitelline membrane

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval(
            'delta_rho**2 * vitelline_K/2')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * _to_3d(
            eptm.vert_df.eval('vitelline_K * delta_rho'))
        grad.columns = ['g' + c for c in eptm.coords]
        return grad, None


class RadialTension(effectors.AbstractEffector):

    dimensions = units.line_tension
    magnitude = 'radial_tension'
    label = 'Apical basal tension'
    element = 'vert'
    specs = {'vert': {'is_active',
                      'height',
                      'radial_tension'}}

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval(
            'height * radial_tension * is_active')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * to_nd(
            eptm.vert_df.eval('radial_tension'), 3)
        grad.columns = ['g' + c for c in eptm.coords]
        return grad, None


def define_mesoderm(sheet, a=145.0, b=40.0, coords=["x", "z", "y"]):
    """
    Define an oval area that will become the mesoderm.
    a: radius on the first-axis
    b: radius on the second-axis
    """
    x, y, z = coords
    x_ = sheet.face_df[x] / b
    y_ = sheet.face_df[y] / a

    radius = x_ ** 2 + y_ ** 2
    height = sheet.face_df[z]

    sheet.face_df["is_mesoderm"] = (radius <= 1) & (height > 0)

    # Remove not active face
    face_not_active = sheet.edge_df[
        ~sheet.edge_df.is_active.astype(bool)].face.unique()
    sheet.face_df.loc[face_not_active, "is_mesoderm"] = False


def define_relaxation_cells(sheet, a=145., b=40.,
                            aa=145., bb=40.,
                            coords=['x', 'z', 'y']):
    """
    Define 2 lines of cells surronding the mesoderm which will be relaxating

    a: radius on the first-axis
    b: radius on the second-axis
    """
    define_mesoderm(sheet, aa, bb)
    x, y, z = coords
    x_ = sheet.face_df[x] / b
    y_ = sheet.face_df[y] / a

    rayon = x_**2 + y_**2
    height = sheet.face_df[z]

    sheet.face_df['is_relaxation'] = ~((rayon <= 1) & (height > 0))
    sheet.face_df['is_relaxation'] = [False if ((not line[1]['is_mesoderm']) & (line[1]['is_relaxation']))
                                      else line[1]['is_relaxation']
                                      for line in sheet.face_df.iterrows()]

    sheet.face_df['is_mesoderm'] = [False if ((line[1]['is_mesoderm']) & (line[1]['is_relaxation']))
                                    else line[1]['is_mesoderm']
                                    for line in sheet.face_df.iterrows()]
    # Remove not active face
    face_not_active = sheet.edge_df[
        ~sheet.edge_df.is_active.astype(bool)].face.unique()
    sheet.face_df.loc[face_not_active, 'is_relaxation'] = False


def initiate_ellipsoid(dataset_path, json_path):
    """
    Create ellipsoid tissue as a sheet with mesodermal cells

    dataset_path: initial hf45 file
    b: json spec file
    """
    dsets = hdf5.load_datasets(dataset_path,
                               data_names=['vert', 'edge', 'face'])

    with open(json_path, 'r+') as fp:
        specs = json.load(fp)

    sheet = Sheet('ellipse', dsets, specs)

    # Modify some initial value
    sheet.settings['threshold_length'] = 1e-3
    sheet.settings['vitelline_space'] = 0.2
    sheet.vert_df['radial_tension'] = 0.
    sheet.settings['lumen_prefered_vol'] = 4539601.384437251
    sheet.settings['lumen_vol_elasticity'] = 3.e-6
    sheet.edge_df.cell = np.nan

    # Define mesoderm and relaxating cells
    define_mesoderm(sheet, 145, 40)
    define_relaxation_cells(sheet, 140, 33, 145, 40)

    return sheet
