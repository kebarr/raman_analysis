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
        self.len_x = len(df.loc[df["x"] == df.x[0]])
        self.len_y = int(len(df)/float(self.len_x))
        self.match_image = np.zeros((self.len_x, self.len_y), np.uint8)

    def add_value_to_image(self, i, con):
        x = int(i)/int(self.len_y)
        y = i%(x*self.len_y)
        self.match_image[x, y] = [0, 0, 255, np.uint8(con*0.01*255)]

    def add_image(self, image_filename, testing = True):
        self.im = Image.open(image_filename)
        np_im = np.array(self.im)
        im_shape = np_im.shape
        im_x = im_shape[0]
        im_y = im_shape[1]
        if not testing:
            if im_x != self.len_x:
                raise ValueError("image x dimension, %d, does not match data x dimension, %d" % (im_x, self.len_x))
            if im_y != self.len_y:
                raise ValueError("image y dimension, %d, does not match data y dimension, %d" % (im_y, self.len_y))

    # need to ask Andy about this
    def show_matches_on_image(self, output_filename):
        final_match_image = Image.fromarray(self.match_image)
        print(np.array(self.im).shape, self.match_image.shape)
        result = Image.blend(self.im, final_match_image, 0.5)
        result.save(output_filename, "PNG")
        # if this doesn't look good, try:
        #background.paste(img, (0, 0), img)
        #background.save('how_to_superimpose_two_images_01.png',"PNG")



