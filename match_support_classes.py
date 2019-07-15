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

    def add_match(self, index, confidence):
        self.matches.append((index, confidence))
        if confidence > self.high_thresh:
            self.high_confidence.append(len(self.matches) -1)
            self.med_confidence.append(len(self.matches) -1)    
        elif confidence > self.med_thresh:
            self.med_confidence.append(len(self.matches) -1)
        

class MatchImage(object):
    def __init__(self, df):
        # not really any restrictions on size/shape of image so can't really do any sanity checks here.
        self.len_x = len(df.loc[df["x"] == df.x[0]])
        self.len_y = int(len(df)/float(self.len_x))
        #self.match_image = np.zeros((self.len_x, self.len_y, 4), np.uint8)

    def add_image(self, image_filename, testing = True):
        # in Andy's code, it looks like he just resizes to make both the same dimensions!!
        im = Image.open(image_filename)
        im = im.convert("RGBA")
        self.im = im.resize((self.len_x+1, self.len_y+1))
        self.im_array = np.array(self.im)


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


