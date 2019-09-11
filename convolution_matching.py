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
import pickle
import math
from scipy.linalg import solveh_banded
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


deriv_order = 2
# Compute the fixed derivative of identity (D).
d = np.zeros(deriv_order*2 + 1, dtype=int)
d[deriv_order] = 1
d = np.diff(d, n=deriv_order)
n = 951
k = len(d)
# fiddle with this carefully
s = float(1e6) 

# Here be dragons: essentially we're faking a big banded matrix D,
# doing s * D.T.dot(D) with it, then taking the upper triangular bands.
diag_sums = np.vstack([
    np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
    for i in range(1, k+1)])
upper_bands = np.tile(diag_sums[:,-1:], n)
upper_bands[:,:k] = diag_sums
for i,ds in enumerate(diag_sums):
    upper_bands[i,-i-1:] = ds[::-1][:i+1]
upper_bands = upper_bands

class WhittakerSmoother(object):
  def __init__(self, signal, deriv_order=1):
    self.y = signal


  def smooth(self, w):
    foo = upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

#https://gist.github.com/perimosocordiae/efabc30c4b2c9afd8a83
# try without smoothing... doesn't work... smooth at beginning or end end?
def als_baseline(intensities, asymmetry_param=0.05, max_iters=3, conv_thresh=1e-5, verbose=False):
    '''Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                        Setting p=1 is effectively a hinge loss.
    '''
    smoother = WhittakerSmoother(intensities)
    # Rename p for concision.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])
    for i in range(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p*mask + (1-p)*(~mask)
        conv = np.linalg.norm(new_w - w)
        if conv < conv_thresh:
            break
        w = new_w
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
        self.template =  np.array(self.material.template)[90:-105]
        self.template /= np.sum(self.template)
        self.matches = Matches(filename, material_name)
        print(self.subtract_baseline)
        if os.path.exists(self.pickle_filename):
            self.load()
        else:
            self.load_data()
            self.find_matches()
            self.write_to_file()

    def subtract_baseline_data(self):
        baseline_filename = self.data_filename.split('.csv')[0] + '_baselined.csv'
        index_to_compare = np.random.randint(0, len(self.spectra))
        self.random_sample_compare_before_subtract_baseline = copy.deepcopy(self.spectra[index_to_compare])
        print("subtracting baselines.... please wait!!!!!")
        for i in range(len(self.spectra)):
            self.spectra[i] = self.spectra[i] - als_baseline(self.spectra[i])
        #print("baseline subtracted, writing result to file: %s " % (baseline_filename))
        self.random_sample_compare_after_subtract_baseline = self.spectra[index_to_compare]

    def load(self):
        print("loading from pickle")
        f = open(self.pickle_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict)


    def write_to_file(self):
        f = open(self.pickle_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load_data(self):
        # td.columns is the raman wavelength
        fname = self.data_filename
        print("reading data from %s to locate %s" % (fname, self.material_name))
        data= pd.read_csv(fname, sep='\t', encoding='utf-8')
        data = data.rename(columns={'Unnamed: 0' : 'x', 'Unnamed: 1' : 'y'})
        data = data.dropna()
        # should be better way to do this, but i can't find it
        shifts = np.array([0 for i in range(len(data.columns[2:]))])
        self.spectra = data.values[:,2:]
        self.positions = list(zip(data.x, data.y)) # can probably change this to unnamed 0 and 1 to cut out rename
        for i, col in enumerate(data.columns[2:]):
            shifts[i] = float(col)
        self.shifts = shifts
        #td.x is x coord
        #td.iloc[0][2:] is just data in column 0 (indexes 0 and 1 are x and y coordinates)
        #td.columns[index] is wavelength at position index
        if self.subtract_baseline == True:
           self.subtract_baseline_data()
        self.spectra = np.apply_along_axis(scipy.ndimage.median_filter, 1, self.spectra, 8)
        print("successfully loaded data")
        # not really any restrictions on size/shape of image so can't really do any sanity checks here.
        self.len_x_0 = len(data.loc[data["x"] == data.x[0]])
        self.x_0 = data.x[0]
        self.y_0 = data.y[0]
        self.x_max = data.x[len(data)-1]
        self.y_max = data.y[len(data)-1]
        self.len = len(data)

    def find_indices_of_peak_shifts(self):
        ##TODO - THIS ASSUMES TWO PEAKS!!! - just make a list and append pairs
        lower_bound_1 = self.material.peaks[0][0]
        upper_bound_1 = self.material.peaks[0][1]
        lower_bound_2 = self.material.peaks[1][0]
        upper_bound_2 = self.material.peaks[1][1]
        cond = ((self.shifts > lower_bound_1) & (self.shifts < upper_bound_1)) | ((self.shifts > lower_bound_2) & (self.shifts < upper_bound_2))
        self.peak_indices = np.where(cond)
        if len(self.peak_indices) == 0:
            raise ValueError("Shifts of data set do not include expected peak shifts")
        # to rule out possibility of getting other d peak and weird stuff at beginning,
        # do +- 200 if powwible
        print(self.peak_indices)
        self.lowest_index = self.peak_indices[0][0]
        self.highest_index = self.peak_indices[0][-1] +25
        print("li: ", self.lowest_index)
        print("hi: ", self.highest_index)
        # guess at reasonable max width
        self.max_width = self.highest_index - self.lowest_index



    def prepare_data(self, index):
        d = np.copy(self.spectra[index])
        d_final = d[self.lowest_index:self.highest_index]
        d_final /= np.sum(d_final)
        return d_final
     

    # TODO COMPARE PERFORMANCE OF DOING THIS FIRST AND CONVOLUTION SECOND, OR OTHER WAY ROUND
    def check_match(self, index):
        peak_start = self.peak_indices[0][0]
        peak_end = self.peak_indices[0][-1]
        spectrum = self.spectra[index]
        # check proposed match by comaring mean of peak region to mean of non peak region
        # this assumes peaks are close enough together to be treated as one block
        max_peaks = np.partition(spectrum[peak_start:peak_end], -3)[-3] # second largest, to avoid cosmic rays being max
        # cut off first bit cos there's some weirdness in Cyrills data.
        mean_non_peaks = (np.mean(spectrum[200:self.peak_indices[0][0]]) + np.mean(spectrum[self.peak_indices[0][-1]:]))*0.5 + 50
        stdev_all = np.std(spectrum)
        mean_peaks = np.mean(spectrum[peak_start:peak_end])
        stdev_peaks = np.std(spectrum[peak_start:peak_end])
        mean_all = np.mean(spectrum)
        stdev_non_peaks = np.std(np.concatenate([spectrum[200:self.peak_indices[0][0]], spectrum[self.peak_indices[0][-1]:]]))
        #if max_peaks >
        if max_peaks > mean_non_peaks+3*stdev_non_peaks:
            confidence = 15*(mean_peaks - mean_non_peaks)/stdev_non_peaks
            confidence = confidence if confidence < 100 else 100
            return True, confidence
        else:
            return False, 0

    def is_match(self, index):
        to_match = self.prepare_data(index)
        conv = scipy.signal.fftconvolve(to_match, self.template)
        #print("conv max: ", np.max(conv))
        if np.max(conv)> 0.0065:# and np.where(conv == np.max(conv))[0][0] < 129 and  np.where(conv == np.max(conv))[0][0] > 120:
            peaks = scipy.signal.find_peaks(to_match, prominence=0.005, width=8)
            peak1_int = peaks[0][np.where(peaks[0] >10)]
            peak2_int = peaks[0][np.where(peaks[0] >70)]
            peak1_final = peak1_int[np.where(peak1_int <50)]
            peak2_final = peak2_int[np.where(peak2_int <130)]
            if len(peak1_final) != 0 and len(peak2_final) != 0:
                if to_match[peak1_final[0]] > to_match[peak2_final[0]]:             
                    return True, conv, [to_match[peak1_final[0]], to_match[peak2_final[0]]]
            return False, conv, peaks
        return False, [], []


    def find_matches(self):
        partial = 0
        self.find_indices_of_peak_shifts()
        number_locations = len(self.spectra)
        print("Searching %d locations for %s" % (number_locations, self.material_name))
        update_flag = int(number_locations/25) # how often to update user
        check_m = 0
        for i in range(number_locations):
            potential_match, con = self.check_match(i)
            if potential_match:
                check_m += 1
                match, conv, peak_data = self.is_match(i)
                if match == True:
                    self.matches.add_match(self.material, con, self.spectra[i], peak_data, self.positions[i][0], self.positions[i][1], i)
                if len(conv) != 0:
                    partial += 1
            if i%update_flag == 0:
                print("Tested %d locations, found %d matches" % (i, len(self.matches.matches)))
        print("%d spectra made it though initial filtering " % (check_m))
        print("Finished finding matches, found %d locations potentially positive for %s, %d partial matches" % (len(self.matches.matches), self.material_name, partial))

    def get_condifence_matches(self, thresh='medium'):
        if thresh=='medium':
            return [self.matches.matches[match] for match in self.matches.med_confidence]
        elif thresh=='high':
            return [self.matches.matches[match] for match in self.matches.high_confidence]
        
    def get_high_confidence_matches(self):
        return self.get_condifence_matches('high')

    def get_med_confidence_matches(self):
        return self.get_condifence_matches()

    def overlay_match_positions(self, bitmap_filename, output_filename, confidence="medium"):
        mi = MatchImage(self.x_0, self.y_0, self.x_max, self.y_max)
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

    def overlay_match_positions_blank(self, output_filename, confidence="medium"):
        mi = MatchImage(self.x_0, self.y_0, self.x_max, self.y_max)
        mi.create_blank_image(self.x_max-self.x_0, self.y_max-self.y_0)
        if confidence == "medium":
            matches = self.get_med_confidence_matches()
        elif confidence == "high":
            matches = self.get_high_confidence_matches()
        else:
            matches = self.matches.matches
        # matches is (match_index, confidence score)
        for match in matches:
            mi.add_value_to_image(match)
        mi.save_image(output_filename)


