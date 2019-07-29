import os
import math
import copy
import random
import csv
import string
import collections
import numpy as np
import pandas as pd
from scipy import optimize, signal
import scipy
from sklearn import preprocessing
import collections
import cPickle

from PIL import Image
from match_support_classes import Matches, MatchImage


def read_template(name):
        filename = name + '.csv'
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            t = []
            for row in reader:
                for item in row:
                   t.append(float(item))
        return t

material = collections.namedtuple('material', 'name peaks template')
graphene_oxide= material(name='graphene_oxide', peaks=[(1250, 1450), (1500, 1700)], template=read_template('matching_templates/graphene_oxide'))
materials = {'graphene_oxide': graphene_oxide}

# function to baseline data
def baseline_als(y, lam=10**6, p=0.01, niter=10):
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    # work out what this actuallly does
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# Convolution-based matching for Raman spectral analysis- identify peaks representitive of specific materials
# Input: raman spectral data
# Each desired material must have a profile, this includes a template and the positions of peaks

# refactor to avoid caching large amounts of data: 
class FindMaterial(object):
    def __init__(self, filename, material_name, subtract_baseline=False):
        self.data_filename = filename
        self.pickle_filename = filename.split(".")[0] + ".pickle"
        self.subtract_baseline = subtract_baseline
        self.material_name = material_name
        self.material = materials[self.material_name]
        self.matches = Matches(filename, material_name)
        if os.path.exists(self.pickle_filename):
            self.load()
        else:
            data = self.load_data()
            self.find_matches(data)
            self.write_to_file()

    def subtract_baseline_data(self, data):
        baseline_filename = self.data_filename.split('.csv')[0] + '_baselined.csv'
        index_to_compare = np.random.randint(0, len(data))
        self.random_sample_compare_before_subtract_baseline = copy.deepcopy(data.iloc[index_to_compare])
        print("subtracting baselines.... please wait!")
        for i in range(len(data)):
            # painful but simple way is prohibitively slow
            new = np.concatenate([np.array([data.iloc[i].x]), np.array([data.iloc[i].y]), np.array(data.iloc[i][2:] - baseline_als(data.iloc[i][2:]))])
            data.loc[i] = new
        print("baseline subtracted, writing result to file: %s " % (baseline_filename))
        self.random_sample_compare_after_subtract_baseline = data.iloc[index_to_compare]
        data.to_csv(baseline_filename, sep='\t')
        return data

    def load(self):
        print("loading from pickle")
        f = open(self.pickle_filename, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict)


    def write_to_file(self):
        f = open(self.pickle_filename, 'wb')
        cPickle.dump(self.__dict__, f, 2)
        f.close()

    def load_data(self):
        # td.columns is the raman wavelength
        fname = self.data_filename
        print("reading data from %s to locate %s" % (fname, self.material_name))
        data= pd.read_csv(fname, sep='\t', encoding='utf-8')
        data = data.rename(columns={'Unnamed: 0' : 'x', 'Unnamed: 1' : 'y'})
        #td.x is x coord
        #td.iloc[0][2:] is just data in column 0 (indexes 0 and 1 are x and y coordinates)
        #td.columns[index] is wavelength at position index
        if self.subtract_baseline == True:
           data = self.subtract_baseline_data(data)
        # should be better way to do this, but i can't find it
        wavelengths = np.array([0 for i in range(len(data.columns[2:]))])
        for i, col in enumerate(data.columns[2:]):
            wavelengths[i] = float(col)
        self.wavelengths = wavelengths
        print("successfully loaded data")
        # not really any restrictions on size/shape of image so can't really do any sanity checks here.
        self.len_x = len(data.loc[data["x"] == data.x[0]])
        self.len_y = int(len(data)/float(self.len_x))
        self.len = len(data)
        return data

    def find_indices_of_peak_wavelengths(self):
        ##TODO - THIS ASSUMES TWO PEAKS!!! - just make a list and append pairs
        lower_bound_1 = self.material.peaks[0][0]
        upper_bound_1 = self.material.peaks[0][1]
        lower_bound_2 = self.material.peaks[1][0]
        upper_bound_2 = self.material.peaks[1][1]
        cond = ((self.wavelengths > lower_bound_1) & (self.wavelengths < upper_bound_1)) | ((self.wavelengths > lower_bound_2) & (self.wavelengths < upper_bound_2))
        self.peak_indices = np.where(cond)
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



    def prepare_data(self, data, index):
        d = data.iloc[index].values[self.lowest_index:self.highest_index]
        d = d.reshape((len(d), 1))
        min_max_scaler = preprocessing.MinMaxScaler()
        d_scaled = min_max_scaler.fit_transform(d)
        d_final = [d_scaled[x][0] for x in range(0, len(d_scaled))]
        return d_final
     

    # TODO COMPARE PERFORMANCE OF DOING THIS FIRST AND CONVOLUTION SECOND, OR OTHER WAY ROUND
    def check_match(self, data, index):
        peak_start = self.peak_indices[0][0]
        peak_end = self.peak_indices[0][-1]
        spectrum = data.iloc[index]
        # check proposed match by comaring mean of peak region to mean of non peak region
        # this assumes peaks are close enough together to be treated as one block
        max_peaks = np.max(spectrum[peak_start:peak_end]) + 50
        # cut off first bit cos there's some weirdness in Cyrills data.
        mean_non_peaks = (np.mean(spectrum[200:self.peak_indices[0][0]]) + np.mean(spectrum[self.peak_indices[0][-1]:]))*0.5 + 50
        stdev_non_peaks = np.std(np.concatenate([spectrum[200:self.peak_indices[0][0]], spectrum[self.peak_indices[0][-1]:]]))
        # TODO- confidence scores can be high when mean of data is close to 0, even for pretty shitty matches, 
        # try basing off standard deviation near peaks
        #print("mean peaks: ", mean_peaks, " mean non peaks: ", mean_non_peaks, " stdev non peaks ", stdev_non_peaks)
        if max_peaks > mean_non_peaks+5*stdev_non_peaks:# be quite forgiving as cosmic rays etc will mess it up
            print(spectrum.shape)
            peak_data = self.get_peak_heights(mean_non_peaks, stdev_non_peaks, spectrum)
            # calculate how far beyond non peak mean as a confidence measure
            if max_peaks < (2000+mean_non_peaks):
                confidence = (100.*max_peaks)/(2000+mean_non_peaks)
            else:
                confidence = 100
            return True, confidence, peak_data
        else:
            return False, 0, [0]

    def is_match(self, data, index):
        res, con, peak_data = self.check_match(data, index)
        if res:
            to_match = self.prepare_data(data, index)
            template = self.material.template
            conv = scipy.signal.fftconvolve(to_match, template)
            conv_peaks  = scipy.signal.find_peaks(conv, width = [118,self.max_width], prominence = 30)
            if len(conv_peaks[0]) == 0:
                return False, 0, [0], [0]
            elif len(conv_peaks[0]) > 0:
                return True, con, conv_peaks, peak_data
        return False, 0, [0], [0]

    def find_matches(self, data):
        self.find_indices_of_peak_wavelengths()
        number_locations = len(data)
        print("Searching %d locations for %s" % (number_locations, self.material_name))
        update_flag = int(number_locations/25) # how often to update user
        for i in range(number_locations):
            if i%update_flag == 0:
                print("Tested %d locations, found %d matches" % (i, len(self.matches.matches)))
            match, con, conv_peaks, peak_data = self.is_match(data, i)
            if match == True:
                match_position = self.get_match_position(i)
                self.matches.add_match(self.material, con, data.iloc[i], conv_peaks, peak_data, match_position)
        print("Finished finding matches, found %d locations positive for %s" % (len(self.matches.matches), self.material_name))

    def get_match_position(self, i):
        x = int(i)/int(self.len_y)
        if x == 0:
            # avoid division by 0
            y = i
        else:
            y = i%(x*self.len_y)
        return (x, y)

    def get_condifence_matches(self, thresh='medium'):
        print(thresh)
        if thresh=='medium':
            return [self.matches.matches[match] for match in self.matches.med_confidence]
        elif thresh=='high':
            return [self.matches.matches[match] for match in self.matches.high_confidence]
        
    def get_high_confidence_matches(self):
        return self.get_condifence_matches('high')

    def get_med_confidence_matches(self):
        return self.get_condifence_matches()

    def overlay_match_positions(self, bitmap_filename, output_filename, confidence="medium"):
        mi = MatchImage(self.len_x, self.len_y)
        mi.add_image(bitmap_filename)
        if confidence == "medium":
            matches = self.get_med_confidence_matches()
        elif confidence == "high":
            matches = self.get_high_confidence_matches()
        else:
            matches = self.matches.matches
        # matches is (match_index, confidence score)
        for match in matches:
            mi.add_value_to_image(match )
        mi.save_image(output_filename)

    def get_peak_heights(self, mean_non_peaks, stdev_non_peaks, spectrum):
        results = []
        peaks = 0
        for peak in range(len(self.material.peaks)):
            peak_present, peak_max = self.peak_at_location(peak, mean_non_peaks, stdev_non_peaks, spectrum)
            results.append((peak_present, peak_max))
            if peak_present:
                peaks += 1
        return results

    def peak_at_location(self, peak_ind, mean_non_peaks, stdev_non_peaks, spectrum):
        # i need index of start of peak and index of end of peak
        # data has been scaled and we don't know by how much....
        cond = ((self.wavelengths > self.material.peaks[peak_ind][0]) & (self.wavelengths < self.material.peaks[peak_ind][1]))
        peak_indices = np.where(cond)
        peak = spectrum[peak_indices[0][0]:peak_indices[0][-1]]
        peak_max = np.max(peak)
        peak_present = False
        if peak_max > mean_non_peaks+5*stdev_non_peaks:
            peak_present = True
        return peak_present, peak_max

