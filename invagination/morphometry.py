"""Cell morphometry module
"""

from math import atan, asin
from pathlib import Path
from matplotlib.ticker import NullFormatter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from tyssue import Sheet, SheetGeometry, config
from tyssue.io import hdf5
from tyssue.draw.plt_draw import quick_edge_draw, sheet_view


__all__ = [
    'analyse',
    'grided_graph',
    'parse_ply',
    'get_borders',
    'get_morphometry',
    ]


labels = {'ar': 'Aspect ratio',
          'area': 'Area (µm²)',
          'orient': 'Orientation (degrees)',
          'tilt' : 'Angle w/r to LR axis (radians)',
          'x': 'Left/right axis (µm)',
          'y': 'Anterior/posterior axis (µm)',
          'z': 'Ventral/dorsal axis (µm)',}

ranges = {'ar': (1., 3.),
          'orient': (0., 90),
          'area': (20, 100)}


def analyse(data_dir, fname, basename='',
            save=True, plot=True, show=False,
            figures=None, axes=None):

    if fname.endswith('ply'):
        with open(data_dir/fname, 'r+') as fh:
            datasets = parse_ply(fh, read_faces=True)
        borders = get_borders(datasets)
        data_container = datasets

    elif fname.endswith('hf5'):
        sheet = get_ventral_patch(data_dir/fname)
        borders = sheet.upcast_srce(sheet.vert_df[sheet.coords])
        borders['label'] = sheet.edge_df['face']
        data_container = sheet

    centers = get_morphometry(borders)
    centers.to_csv(f'morphometry_{basename}.csv')
    coords = ['x', 'y']
    if not plot:
        return centers

    if figures is None:
        figures = {}
        axes = {}

    for col in ['ar', 'area', 'orient']:
        fig, ax = figures.get(col, None), axes.get(col, None)
        figures[col], axes[col] = grided_graph(
            data_container, centers,
            col, coords, fig, ax)
        axes[col][1].set_title(basename)
        if save:
            figures[col].savefig(f'{basename}_{col}.png', dpi=300)
        if not show:
            plt.close("all")

    return centers, figures, axes


def parse_ply(fh, read_faces=False):
    """Parses a `.ply` file exported from MorphoGraphX to retrieve
    epithelium segmentation.

    Note
    ----
    This is _not_ a generic PLY parser, and will only work for
    headers of the form ::

        ply
        format ascii 1.0
        comment Exported from MorphoGraphX 1.0 1280
        comment Triangle mesh
        element vertex 76801
        property float x
        property float y
        property float z
        property int label
        property float signal
        element face 152496
        property list uchar int vertex_index
        property int label
        end_header

    Parameters
    ----------
    fh : file handle
    read_face : bool, default False
        if `True`, reads the data from the face triangles

    Returns
    -------
    datasets : dict
        The datasets dictionnary is of the form
        `{'vert': vert_df, 'face': face_df}`.
        The vertex Dataframe has ['x', 'y', 'z', 'label', 'signal']
        columns, and the face Dataframe has ['nv', 'v0', 'v1', 'v2', 'label']
        columns, where v0, v1, v2 are indices of the face trianglule vertices.

    """

    columns_v = ['x', 'y', 'z', 'label', 'signal']
    dtypes_v = [float, float, float, int, float]
    columns_f = ['nv', 'v0', 'v1', 'v2', 'label']
    dtypes_f = [int,]*len(columns_f)
    header_length = 0
    for line in fh:
        header_length += 1
        if line.startswith('element vertex'):
            num_v = int(line.split(' ')[-1])
        elif line.startswith('element face') :
            num_f = int(line.split(' ')[-1])
        elif line.startswith('end_header'):
            break

    vert_df = pd.read_csv(fh, sep=' ',
                          nrows=num_v,
                          header=None,
                          names=columns_v,
                          dtype={name: dt for dt, name
                                 in zip(dtypes_v, columns_v)})
    vert_df.index.name = 'vert'

    if not read_faces:
        return {'vert': vert_df}

    # Restart at top of file (because of buffering I guess)
    fh.seek(0)
    face_df = pd.read_csv(fh, sep=' ',
                          nrows=num_f,
                          skiprows=header_length+num_v,
                          header=None,
                          names=columns_f, dtype=int)
    face_df.index.name = 'face'
    return {"vert": vert_df, "face": face_df}


def get_borders(datasets):

    def _border_verts(face):
        sub = datasets['vert']['label'].take(
            face[['v0', 'v1', 'v2']].values.ravel())
        border = sub[sub != face.label.values[0]].index.astype(int)
        return pd.Series(border, name='vert')

    borders = datasets['face'].groupby('label').apply(
        _border_verts).reset_index()[['label', 'vert']]
    ## remove membrane and background
    borders = borders[borders["label"] > 0].drop_duplicates()
    for c in ['x', 'y', 'z']:
        borders[c] = datasets['vert'].loc[borders['vert'].values, c].values

    return borders


def get_morphometry(borders):

    centers = borders.groupby('label')[['x', 'y', 'z']].mean()
    centers.index = centers.index.astype(np.int)
    centers.index.name='label'

    aniso = borders.groupby('label').apply(_get_anisotropies)
    aniso.index = aniso.index.astype(np.int)
    aniso.index.name = 'label'
    centers = pd.concat([centers, aniso], axis=1)
    centers['orient'] = (np.pi/2 - centers['tilt'])*180/np.pi

    return centers


def _poly_area(relative_pos):
    rolled = np.roll(relative_pos, 1, axis=0)

    return (relative_pos[:, 1] * rolled[:, 0]
            - relative_pos[:, 0]* rolled[:, 1]).sum()/2


def _get_anisotropies(verts, coords=['x', 'y', 'z']):
    centered = (verts[coords].values -
                verts[coords].mean(axis=0).values[np.newaxis, :])
    u, s, v = np.linalg.svd(centered,
                            full_matrices=False,
                            compute_uv=True)

    tilt = np.abs(atan(v[1, 0]/v[0, 0]))
    ar = s[0]/s[1]
    aligned = centered @ v.T
    theta = np.arctan2(aligned[:, 1], aligned[:, 0])
    aligned = aligned[np.argsort(theta)]
    area = _poly_area(aligned)

    return pd.Series({'ar': ar, 'tilt': tilt, 'area': area})


def _create_axes_grid():

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_triplot = [left, bottom, width, height]
    rect_scatx = [left, bottom_h, width, 0.2]
    rect_scaty = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(figsize=(10, 10))

    ax_triplot = fig.add_axes(rect_triplot)

    ax_scatx = fig.add_axes(rect_scatx)
    ax_scaty = fig.add_axes(rect_scaty)

    return fig, (ax_triplot, ax_scatx, ax_scaty)


def grided_graph(data_container,
                 centers,
                 col,
                 coords=['x', 'y'],
                 fig=None, axes=None):

    x, y = coords
    if fig is None:
        fig, (ax_triplot, ax_scatx, ax_scaty) = _create_axes_grid()
    else:
        (ax_triplot, ax_scatx, ax_scaty) = axes

    # no labels
    nullfmt = NullFormatter()         # no labels
    ax_scatx.xaxis.set_major_formatter(nullfmt)
    ax_scaty.yaxis.set_major_formatter(nullfmt)


    if isinstance(data_container, Sheet):
        fig, ax_triplot = color_plot_silico(
            sheet=data_container, centers=centers, col=col,
            ax=ax_triplot, coords=coords)
    elif isinstance(data_container, dict):
        fig, ax_triplot = color_plot_vivo(
            datasets=data_container, centers=centers, col=col,
            ax=ax_triplot, coords=coords)
    else:
        raise ValueError('Type of data_container not understood')

    ax_triplot.set_xlim(-120, 120)
    ax_triplot.set_ylim(-120, 120)
    norm = matplotlib.colors.Normalize(vmin=ranges[col][0],
                                       vmax=ranges[col][1],
                                       clip=True)

    ax_scatx.scatter(centers[x], centers[col],
                     c=centers[col], norm=norm,  alpha=0.8)
    ax_scaty.scatter(centers[col], centers[y],
                     c=centers[col], norm=norm,  alpha=0.8)

    ax_scatx.set_xlim(ax_triplot.get_xlim())
    ax_scaty.set_ylim(ax_triplot.get_ylim())

    ax_scatx.set_ylim(ranges[col])
    ax_scaty.set_xlim(ranges[col])

    ax_scatx.set_ylabel(labels[col], fontsize=12)
    ax_scaty.set_xlabel(labels[col], fontsize=12)

    ax_triplot.set_xlabel(labels[x], fontsize=12)
    ax_triplot.set_ylabel(labels[y], fontsize=12)

    return fig, (ax_triplot, ax_scatx, ax_scaty)


def color_plot_vivo(datasets, centers, col,
               ax=None, coords=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    x, y = coords
    dv = datasets['vert']
    df = datasets['face'][datasets['face'].label > 0].copy()
    df[col] = centers.loc[df.label.values, col].values
    norm = matplotlib.colors.Normalize(vmin=ranges[col][0],
                                       vmax=ranges[col][1],
                                       clip=True)
    p = ax.tripcolor(dv[x], dv[y],
                     df[['v0', 'v1', 'v2']],
                     df[col], norm=norm, lw=0.)

    ax.set_aspect('equal')
    #fig.colorbar(p, ax=ax)
    return fig, ax


def color_plot_silico(sheet, centers, col,
                      ax=None, coords=['x', 'y']):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    normed = (centers[col] - ranges[col][0])/(ranges[col][1] - ranges[col][0])
    face_colors = plt.cm.viridis(normed)

    fig, ax = sheet_view(sheet, coords=coords,
                         ax=ax,
                         face={'visible':True,
                               'color': face_colors},
                         vert={'visible':False})
    ax.grid()
    return fig, ax


'''
In silico analysis
'''



def get_ventral_patch(fname):

    sim_dsets = hdf5.load_datasets(fname)
    sheet = Sheet('morph', sim_dsets,
                  config.geometry.flat_sheet())
    to_crop = sheet.cut_out([[-300, 300],
                             [sheet.vert_df.y.max()-22,
                              sheet.vert_df.y.max()+1],
                             [-300, 300]],)

    sheet.remove(to_crop)
    sheet.vert_df[['x', 'y', 'z']] = sheet.vert_df[['x', 'z', 'y']]
    SheetGeometry.update_all(sheet)
    return sheet
