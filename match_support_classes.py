import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Matches(object):
    def __init__(self, filename, material_name, med_thresh=60, high_thresh=68):
        self.filename = filename
        self.material_name = material_name
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh
        self.matches = []
        self.high_confidence = []
        self.med_confidence = []

    def add_match(self, index, confidence, spectrum):
        self.matches.append((index, confidence, spectrum))
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
        #self.match_image = np.zeros((self.len_x, self.len_y, 4), np.uint8)

    def add_image(self, image_filename, testing = True):
        # in Andy's code, it looks like he just resizes to make both the same dimensions!!
        im = Image.open(image_filename)
        im = im.convert("RGBA")
        #resize image leaves slight border around the outside, so increase dimensions so actual image dimensions match
        self.im = im.resize((self.len_x+4, self.len_y+4))
        self.im_array = np.array(self.im)
        print(self.im_array.shape)

    def add_value_to_image(self, i, con):
        x = int(i)/int(self.len_y)
        if x == 0:
            # avoid division by 0
            y = i
        else:
            y = i%(x*self.len_y)
        # just override previous value in image
        self.im_array[x+2, y+1] = [0, 0, 255, np.uint8(con*0.01*255)]

    # need to ask Andy about this
    def save_image(self, output_filename):
        final = Image.fromarray(self.im_array)
        final.save(output_filename, "PNG")


