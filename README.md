# Invagination

![Sample output from the model](data/svg/header.svg)

[![DOI](https://zenodo.org/badge/185421680.svg)](https://zenodo.org/badge/latestdoi/185421680)


This is a mesoderm invagination simulation package for the article:

### Mechanical impact of epithelial−mesenchymal transition on epithelial morphogenesis in Drosophila

Mélanie Gracia, Sophie Theis, Amsha Proag, Guillaume Gay, Corinne Benassayag, and Magali Suzanne https://www.nature.com/articles/s41467-019-10720-0



## Try it with my binder by clicking the badge bellow:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/suzannelab/invagination/master?filepath=notebooks%2FIndex.ipynb)


## Dependencies

- python > 3.6
- tyssue >= 0.3.3


## Installation

This package is based on the [`tyssue`](https://tyssue.readthedocs.org) library and its dependencies.

The recommanded installation route is to use the `conda` package manager. You can get a `conda` distribution for your OS at https://www.anaconda.com/download . Make sure to choose a python 3.6 version. Once you have installed conda, you can install tyssue with:

```bash
$ conda install -c conda-forge tyssue
```

You can then download and install invagination from github:

- with git:

```bash
$ git clone https://github.com/DamCB/invagination.git
$ cd invagination
$ python setup.py install
```

- or by downloading https://github.com/DamCB/invagination/archive/master.zip ,  uncompressing the archive and running `python setup.py install` in the root directory.

## Licence

This work is free software, published under the MPLv2 licence, see LICENCE for details.


&copy; The article authors -- all rights reserved
