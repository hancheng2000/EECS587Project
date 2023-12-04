import numpy as np
import numba
from numba.experimental import jitclass
from numba import float64
from numba.typed import Dict
from numba.core import types
import os
import pandas as pd
import copy

def