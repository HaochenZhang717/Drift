from .custom import Custom
from .ETTh import ETTh
from .ETTm import ETTm
try:
    from .UEA import UEA
except ModuleNotFoundError:
    UEA = None
try:
    from .glounts import GLUONTS
except ModuleNotFoundError:
    GLUONTS = None

from .sine import Sine
from .stock import Stock
from .energy import Energy
from .mujoco import Mujoco
from .air_quality import AirQuality
from .aireadi import AIREADI, AIREADICalorie, AIREADIGlucose
from .verbal_ts import VerbalTS
from .npy_ts import NpyTimeSeries

from .MSL import MSL
from .PSM import PSM
from .SMAP import SMAP
from .SMD import SMD
from .SWAT import SWAT
