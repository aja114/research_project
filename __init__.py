import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.seterr(all="ignore")

from rl_research import algorithms
from rl_research import experiments
from rl_research import tests
