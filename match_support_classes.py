import numpy as np
from PIL import Image

# container for match stuff to improve readability
class Match(object):
    def __init__(self, material, confidence, spectrum, peak_data, x, y):
        self.material = material
        self.confidence = confidence
        self.spectrum = spectrum
        self.peak_data = peak_data
        self.x = x
        self.y = y
        #self.peak_ratio = self.calculate_peak_ratio()

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
    def __init__(self, filename, material, med_thresh=31, high_thresh=70): 
        self.filename = filename
        self.material = material
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh
        self.matches = []
        self.high_confidence = []
        self.med_confidence = []

    def add_match(self, material, confidence, spectrum, peak_data, x, y):
        match = Match(material, confidence, spectrum, peak_data, x, y)
        self.matches.append(match)
        #if match.peak_ratio > 1.05:
        #    self.matches.append(match)
        #    if confidence > self.high_thresh:
        #            self.high_confidence.append(len(self.matches) -1)
        #            self.med_confidence.append(len(self.matches) -1)    
        #    elif confidence > self.med_thresh:
        #            self.med_confidence.append(len(self.matches) -1)

    
        

class MatchImage(object):
    def __init__(self, x_0, y_0, x_max, y_max):
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_max = x_max
        self.y_max = y_max
        self.len_x = np.abs(x_max - x_0)
        self.len_y = np.abs(y_max - y_0)
        print("x0 %d, y0 %d, xmax %d ymax %d len x 0 %d len y 0 %d" % (x_0, y_0, x_max, y_max, self.len_x, self.len_y))

    def add_image(self, image_filename):
        # in Andy's code, it looks like he just resizes to make both the same dimensions!!
        im = Image.open(image_filename)
        im = im.convert("RGBA")
        #resize image leaves slight border around the outside, so increase dimensions so actual image dimensions match
        self.im_array = np.array(im)
        # need to upsample data rather than resize image as otherwise it is unacceptably blurry
        self.x_scale_factor = float(len(self.im_array))/(self.len_x+1)
        self.y_scale_factor = float(len(self.im_array[0]))/(self.len_y+1)
        print("x scale foctore: %f, y scale factor %f" % (self.x_scale_factor, self.y_scale_factor))

    def create_blank_image(self, len_x, len_y):
        print("creating blank image: %d, %d " % (len_x, len_y))
        # need to work out dimensions of image
        self.x_scale_factor = float(len_x)/(self.len_x+1)
        self.y_scale_factor = float(len_y)/(self.len_y+1)
        self.im_array = np.array([[[0, 0, 0, 255] for i in range(int(len_y))] for j in range(int(len_x))])
        print(self.im_array.shape)

    def add_value_to_image(self, match):
        con = match.confidence
        # scale it manually to increase contrast
        if con > 60:
            contrast = 100
        elif con > 50:
            contrast = 20
        elif con > 35:
            contrast = 10 
        elif con > 20:
            contrast = 5
        else:
            contrast = 1 
        # just override previous value in image
        self.im_array[int(self.x_scale_factor*(match.x- self.x_0)), int(self.y_scale_factor*(match.y-self.y_0))] = [57, 255, 20, np.uint8(con*0.01*255)]
        x_to_add = int(self.x_scale_factor+2)
        y_to_add = int(self.y_scale_factor+2)
        if int(self.x_scale_factor*(match.x- self.x_0)) + x_to_add -1 < len(self.im_array):
            if int(self.y_scale_factor*(match.y- self.y_0)) + y_to_add -1 < len(self.im_array[0]):
                for i in range(x_to_add):
                    for j in range(y_to_add):
                        # as scaled up, need to colous other pixels according to scale factor
                        self.im_array[int(self.x_scale_factor*(match.x- self.x_0))+i, int(self.y_scale_factor*(match.y-self.y_0))+j] = [57, 255, 20, np.uint8(con*0.01*255)]
                        self.im_array[int(self.x_scale_factor*(match.x- self.x_0))-i, int(self.y_scale_factor*(match.y-self.y_0))-j] = [57, 255, 20, np.uint8(con*0.01*255)]
                        self.im_array[int(self.x_scale_factor*(match.x- self.x_0))+i, int(self.y_scale_factor*(match.y-self.y_0))-j] = [57, 255, 20, np.uint8(con*0.01*255)]
                        self.im_array[int(self.x_scale_factor*(match.x- self.x_0))-i, int(self.y_scale_factor*(match.y-self.y_0))+j] = [57, 255, 20, np.uint8(con*0.01*255)]


    # need to ask Andy about this
    def save_image(self, output_filename):
        final = Image.fromarray(self.im_array.astype(np.uint8))
        final.save(output_filename, "PNG")


