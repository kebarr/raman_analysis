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
#import collections


def read_template(name):
        filename = name + '.csv'
        with open(filename, 'r') as f:
            writer = csv.reader(f)
            t = []
            for row in writer:
                for item in row:
                   t.append(float(item))
        return t

material = collections.namedtuple('material', 'name peaks template')
graphene_oxide= materials(name='graphene_oxide', peaks=[(1250, 1450), (1500, 1700)], template=read_template('templates/graphene_oxide'))
materials = {'graphene_oxide': graphene_oxide}

# Convolution-based matching for Raman spectral analysis- identify peaks representitive of specific materials
# Input: raman spectral data
# Each desired material must have a profile, this includes a template and the positions of peaks

# think this is redundant
#class Material(object):
#    def __init__(self, name):
#        self.name = string.lower(name) # name of material, will eventually be provided to user via drop down menu or similar
#        self.material = materials[self.name] # template of the peaks

class FindMaterial(object):
    def __init__(self, material_name, data_filename):
        self.material_name = string.lower(material_name)
        self.data_filename = data_filename
        self.material = materials[self.material_name]
        self.load_data()

    def load_data(self):
        fname = self.data_filename
        data= pd.read_csv(testfile_name, sep='\t', encoding='utf-8')
        # td.columns is the raman wavelength
        data = td.rename(columns={'Unnamed: 0' : 'x', 'Unnamed: 1' : 'y'})
        #td.x is x coord
        #td.iloc[0][2:] is just data in column 0 (indexes 0 and 1 are x and y coordinates)
        #td.columns[index] is wavelength at position index
        self.data = data
    