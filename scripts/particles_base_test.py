#!/usr/bin/env python

# Early test of usage of Particles python library in ROS for bayesian estimation
# The only version which can be used is the legacy 0.1 which supports python 2, mandatory for ROS

import warnings; warnings.simplefilter('ignore')  # Skip warnings, like the one for quasi-monte-carlo features 

from matplotlib import pyplot as plt
import numpy as np 
import seaborn as sb 

import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import MomentsCollector  # In newer version is called 'Moments'
