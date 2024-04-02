__version__ = '0.0.7'
# __version__ = '0.0.8'
# __version__ = '0.1.0' ERC version 

import sys

major, minor = sys.version_info.major, sys.version_info.minor

if major < 3:
    sys.exit("Sorry, Python 2 is not supported. You need Python >= 3.6 for "+__package__+".")
elif minor < 6:
    sys.exit("Sorry, You need Python >= 3.6 for "+__package__+".")
else:
    print(">>>>>>>>>>---"+__package__+"---<<<<<<<<<<")

from neu4mes.main import Neu4mes

from neu4mes.relation import Relation, NeuObj, merge
from neu4mes.input import Input, ControlInput 
from neu4mes.output import Output 

from neu4mes.linear import Linear, LinearBias
from neu4mes.localmodel import LocalModel
from neu4mes.activation import Relu 
from neu4mes.arithmetic import Sum, Subtract, Minus, Square
from neu4mes.trigonometric import Sin, Cos, Tan

import os, os.path
#from pprint import pp, pprint
import numpy as np