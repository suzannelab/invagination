"""
Contractility gradient behavior in a 2.5D sheet

"""

import random
import numpy as np
import math

from tyssue import SheetGeometry
from tyssue.behaviors.sheet_events import (contract,
                                           ab_pull)
from invagination.delamination import (neighbors_contraction,
                                       relaxation,
                                       contract2)

