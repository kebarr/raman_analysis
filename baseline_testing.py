import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from PIL import Image


lam=10**6
L = 951
D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
w = np.ones(L)
W = scipy.sparse.spdiags(w, 0, L, L)
Z = W + lam * D.dot(D.transpose())

def baseline_als_define_outside(y, lam=10**6, p=0.01, niter=10, L=L, D=D, w=w, W=W, Z=Z):
    lam=10**6
    L = 951
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    for i in range(niter):
        z = scipy.sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z



def baseline_als_original(y, lam=10**6, p=0.01, niter=10):
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

import pandas as pd 





def test_baseline_new(data, baseline_func):
    data_arr = np.array(data)
    for i in range(len(data)):
        new = np.concatenate([np.array([data.iloc[i].x]), np.array([data.iloc[i].y]), data_arr[i][2:] - baseline_func(data_arr[i][2:])])
        data.iloc[i] = new
    return data

data= pd.read_csv('uploads/raw_example.csv', sep='\t', encoding='utf-8')
data = data.rename(columns={'Unnamed: 0' : 'x', 'Unnamed: 1' : 'y'})
# made no difference whatsoever!!! so there was a bug that meant it couldn't work
res = test_baseline_new(data, baseline_als_define_outside)

# this does soething, changes original data though
res = test_baseline_new(data, baseline_als_original)

# awk -F',' '{s= $1 ", " $2 ","; for (i=7; i <=NF;i++) s= s $i ", "; print s}'
# argh.... these are with baseline already subtracted
#data_matches= pd.read_csv('med_conf_shifts_only.csv', encoding='utf-8', header=None)
#data_matches = data_matches.drop(columns=[954, 953])

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

wrapped = wrapper(test_baseline_new, data, baseline_als_original)
timeit.timeit(wrapped, number=10) 


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in xrange(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

from scipy.linalg import solveh_banded
#https://gist.github.com/perimosocordiae/efabc30c4b2c9afd8a83
def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,max_iters=5, conv_thresh=1e-5, verbose=False):
    '''Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                        Setting p=1 is effectively a hinge loss.
    '''
    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p for concision.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])
    for i in xrange(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p*mask + (1-p)*(~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print i+1, conv
        if conv < conv_thresh:
            break
        w = new_w
    else:
        print 'ALS did not converge in %d iterations' % max_iters
    return z

wrapped = wrapper(test_baseline_new, data, als_baseline)
timeit.timeit(wrapped, number=10) # 413 seconds for 10 iteractions

from scipy.signal import savgol_filter

def func(x, a, b, c, d, e, f, g):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6


def baseline_poly(data):
    x=[i for i in range(len(data))]
    g, f, e, d, c, b, a = np.polyfit(x, data, 6)
    y_fitted = np.array([func(xi, a, b, c, d, e, f, g) for xi in x])
    return data - y_fitted
    

def baseline_poly_smooth(data):
    x=[i for i in range(len(data))]
    g, f, e, d, c, b, a = np.polyfit(x, data, 6)
    y_fitted = np.array([func(xi, a, b, c, d, e, f, g) for xi in x])
    y_baselined = data -  y_fitted
    return savgol_filter(y_baselined, 51, 3)
    
    

wrapped = wrapper(test_baseline_new, data, baseline_poly)
timeit.timeit(wrapped, number=10)

wrapped = wrapper(test_baseline_new, data, baseline_poly_smooth)
timeit.timeit(wrapped, number=10)
