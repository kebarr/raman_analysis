import os
import math
import random
import csv
import string

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize, signal
import scipy
from sklearn import preprocessing


# Convolution-based matching for Raman spectral analysis- identify peaks representitive of specific materials
# Input: raman spectral data
# Each desired material must have a profile, this includes a template and the positions of peaks

class Material(object):
    def __init__(self, string.lower(name):
        self.name = name # name of material, will eventually be provided to user via drop down menu or similar
        # TODO: 
        self.template = template # template of the peaks

    def read_template(self, name):
        with open('template.csv', 'r') as f:
            writer = csv.reader(f)
            t = []
            for row in writer:
                for item in row:
                   t.append(float(item))