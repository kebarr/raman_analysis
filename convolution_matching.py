import os
import math
import random
import csv
import string
import collections
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
graphene_oxide= material(name='graphene_oxide', peaks=[(1250, 1450), (1500, 1700)], template=read_template('templates/graphene_oxide'))
materials = {'graphene_oxide': graphene_oxide}

# Convolution-based matching for Raman spectral analysis- identify peaks representitive of specific materials
# Input: raman spectral data
# Each desired material must have a profile, this includes a template and the positions of peaks

class FindMaterial(object):
    def __init__(self, material_name, data_filename):
        self.material_name = material_name
        self.data_filename = data_filename
        self.material = materials[self.material_name]
        self.load_data()

    def load_data(self):
        fname = self.data_filename
        print("reading data from %s to locate %s" % (fname, self.material_name))
        data= pd.read_csv(fname, sep='\t', encoding='utf-8')
        # td.columns is the raman wavelength
        data = data.rename(columns={'Unnamed: 0' : 'x', 'Unnamed: 1' : 'y'})
        #td.x is x coord
        #td.iloc[0][2:] is just data in column 0 (indexes 0 and 1 are x and y coordinates)
        #td.columns[index] is wavelength at position index
        self.data = data
        # should be better way to do this, but i can't find it
        wavelengths = np.array([0 for i in range(len(data.columns[2:]))])
        for i, col in enumerate(data.columns[2:]):
            wavelengths[i] = float(col)
        self.wavelengths = wavelengths
        print("successfully loaded data")

    def find_indices_of_peak_wavelengths(self):
        ##TODO - THIS ASSUMES TWO PEAKS!!! - just make a list and append pairs
        lower_bound_1 = self.material.peaks[0][0]
        upper_bound_1 = self.material.peaks[0][1]
        lower_bound_2 = self.material.peaks[1][0]
        upper_bound_2 = self.material.peaks[1][1]
        #print(self.wavelengths)
        cond = ((self.wavelengths > lower_bound_1) & (self.wavelengths < upper_bound_1)) | ((self.wavelengths > lower_bound_2) & (self.wavelengths < upper_bound_2))
        self.peak_indices = np.where(cond)
        print(self.peak_indices)
        if len(self.peak_indices) == 0:
            raise ValueError("Wavelengths of data set do not include expected peak wavelengths")
        # to rule out possibility of getting other d peak and weird stuff at beginning,
        # do +- 200 if powwible
        if self.peak_indices[0][0] > 201:
            self.lowest_index = self.peak_indices[0][0] - 200
        else:
            self.lowest_index = self.peak_indices[0][0]
        if self.peak_indices[0][-1] < len(self.wavelengths) -400:
            self.highest_index = self.peak_indices[0][-1] +400
        elif self.peak_indices[0][-1] < len(self.wavelengths) -200:
            self.highest_index = self.peak_indices[0][-1] +200
        else:
            self.highest_index = self.peak_indices[0][-1]
        print(self.lowest_index, self.highest_index)
        # guess at reasonable max width
        self.max_width = self.highest_index - self.lowest_index
        print("max width: ", self.max_width)




    def prepare_data(self, index):
        d = self.data.iloc[index].values[self.lowest_index:self.highest_index]
        d = d.reshape((len(d), 1))
        min_max_scaler = preprocessing.MinMaxScaler()
        d_scaled = min_max_scaler.fit_transform(d)
        d_final = [d_scaled[x][0] for x in range(0, len(d_scaled))]
        return d_final
    
    def is_match(self, index):
        # prepare data
        to_match = self.prepare_data(index)
        template = self.material.template
        conv = scipy.signal.fftconvolve(to_match, template)
        conv_peaks  = scipy.signal.find_peaks(conv, width = [118,self.max_width], prominence = 30)
        if len(conv_peaks[0]) == 0:
            return False
        elif len(conv_peaks[0]) > 0:# & (conv_peaks[0][0] < 270) & (conv_peaks[0][0] > 230):
            #print(index, conv_peaks)
            if self.check_match(index):
                #print("match at ", index, " accepted, conv peaks: ", conv_peaks)
                #plt.plot(conv)
                #plt.show()
                #self.data.iloc[index].plot()
                #plt.show()
                #print("conv match: ", conv[conv_peaks[0][0]-15:conv_peaks[0][0] + 15])
                # what if there are multiple convolution peaks? 
                #### TODO- the bounds will actually be determined by the data
                # position in convolution is actual match position + size(template)
                # so to position of convolution peak should be position in data - template length
                return True
        return False

    # TODO COMPARE PERFORMANCE OF DOING THIS FIRST AND CONVOLUTION SECOND, OR OTHER WAY ROUND
    def check_match(self, index):
        # check proposed match by comaring mean of peak region to mean of non peak region
        # this assumes peaks are close enough together to be treated as one block
        mean_peaks = np.mean(self.data.iloc[index][self.peak_indices[0][0]:self.peak_indices[0][-1]]) 
        #print("mean peaks for index %d: %f", (index, mean_peaks))
        # cut off first bit cos there's some weirdness in Cyrills data.
        mean_non_peaks = np.mean(self.data.iloc[index][200:self.peak_indices[0][0]]) + np.mean(self.data.iloc[index][self.peak_indices[0][-1]:])
        stdev_non_peaks = np.std(self.data.iloc[index][200:self.peak_indices[0][0]]) + np.mean(self.data.iloc[index][self.peak_indices[0][-1]:])
        #print("mean non peaks: %f stdev non peaks: %f", (mean_non_peaks, stdev_non_peaks))
        if mean_peaks > mean_non_peaks+stdev_non_peaks:# be quite forgiving as cosmic rays etc will mess it up
            return True
        else:
            return False

    def find_matches(self):
        self.find_indices_of_peak_wavelengths()
        number_locations = len(self.data)
        print("Searching %d locations for %s" % (number_locations, self.material_name))
        update_flag = int(number_locations/100) # how often to update user
        matches = []
        for i in range(number_locations):
            if i%update_flag == 0:
                print("Tested %d locations, found %d matches" % (i, len(matches)))
            match = self.is_match(i)
            if match == True:
                matches.append(i)
                #self.data.iloc[i].plot()
                #åplt.show()
        print("Finished finding matches, found %d locations positive for %s" % (len(matches), self.material_name))
        self.matches = matches

    def find_match_positions(self):
        match_positions = []
        for match in self.matches:
            match_scaled = match + self.lowest_index
            match_positions.append(self.data.x[match_scaled], self.data.y[match_scaled])
        self.match_positions = []

    def plot_matches(self):
        for match in self.matches:
            self.data.iloc[match].plot()
            plt.show()

