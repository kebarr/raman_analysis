import numpy as np
from PIL import Image

# container for match stuff to improve readability
class Match(object):
    def __init__(self, material, confidence, spectrum, conv_peaks, peak_data, location):
        self.material = material
        self.confidence = confidence
        self.spectrum = spectrum
        self.conv_peaks = conv_peaks
        self.peak_data = peak_data
        self.x = location[0]
        self.y = location[1]
        self.peak_ratio = self.calculate_peak_ratio()

    def calculate_peak_ratio(self):
        number_peaks = len(self.material.peaks)
        peaks_present = 0
        peak_ratio = 0
        for peak in self.peak_data:
            if peak[0]:
                peaks_present +=1
        if number_peaks == 2 and peaks_present == number_peaks:
            # not sure how to do this if more than 2 peaks
            peak_ratio = np.float(self.peak_data[0][1])/self.peak_data[1][1]
        return peak_ratio

    def to_dict(self):
        return {'confidence': self.confidence, 'spectrum': self.spectrum, 'x' : self.x, 'y': self.y, 'peak_ratio':self.peak_ratio, 'peak_data': self.peak_data}

class Matches(object):
    def __init__(self, filename, material, med_thresh=35, high_thresh=50): 
        self.filename = filename
        self.material = material
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh
        self.matches = []
        self.high_confidence = []
        self.med_confidence = []

    def add_match(self, material, confidence, spectrum, conv_peaks, peak_data, location):
        self.matches.append(Match(material, confidence, spectrum, conv_peaks, peak_data, location))
        if confidence > self.high_thresh:
            self.high_confidence.append(len(self.matches) -1)
            self.med_confidence.append(len(self.matches) -1)    
        elif confidence > self.med_thresh:
            self.med_confidence.append(len(self.matches) -1)

    
        

class MatchImage(object):
    def __init__(self, x, y):
        self.len_x = x
        self.len_y = y
        print(x, y)

    def add_image(self, image_filename):
        # in Andy's code, it looks like he just resizes to make both the same dimensions!!
        im = Image.open(image_filename)
        im = im.convert("RGBA")
        #resize image leaves slight border around the outside, so increase dimensions so actual image dimensions match
        self.im = im.resize((self.len_x+4, self.len_y+4))
        self.im_array = np.array(self.im)
        print(self.im_array.shape)

    def add_value_to_image(self, match):
        con = match.confidence
        # scale it manually to increase contrast
        if con > 70:
            contrast = 100
        elif con > 60:
            contrast = 80
        elif con > 45:
            contrast = 65 
        elif con > 30:
            contrast = 40
        else:
            contrast = 10 
        # just override previous value in image
        self.im_array[match.x+2, match.y+1] = [255, 128, 0, np.uint8(con*0.01*255)]

    # need to ask Andy about this
    def save_image(self, output_filename):
        final = Image.fromarray(self.im_array)
        final.save(output_filename, "PNG")


