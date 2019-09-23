import numpy as np
from PIL import Image

# container for match stuff to improve readability
class Match(object):
    def __init__(self, material, confidence, spectrum, peak_data, x, y, i):
        self.material = material
        self.confidence = confidence
        self.spectrum = spectrum
        self.peak_data = peak_data
        self.x = x
        self.y = y
        self.i = i
        self.calculate_peak_ratio()

    def calculate_peak_ratio(self):
        self.peak_ratio  = np.float(self.peak_data[0])/self.peak_data[1]

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

    def add_match(self, material, confidence, spectrum, peak_data, x, y, i):
        match = Match(material, confidence, spectrum, peak_data, x, y, i)
        self.matches.append(match)
        if confidence > self.high_thresh:
            self.high_confidence.append(len(self.matches) -1)
            self.med_confidence.append(len(self.matches) -1)    
        elif confidence > self.med_thresh:
            self.med_confidence.append(len(self.matches) -1)

    
        
### TODO- this crashes for x0 = -956, y0=-438, xmax=-840, ymax=-128, len x =123, len y = 310,
# index 367 is out of bounds for array size 351
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
        print("x scale foctor: %f, y scale factor %f" % (self.x_scale_factor, self.y_scale_factor))

    def create_blank_image(self, len_x, len_y):
        print("creating blank image: %d, %d " % (len_x, len_y))
        # need to work out dimensions of image
        self.x_scale_factor = float(len_x)/(self.len_x+1)
        self.y_scale_factor = float(len_y)/(self.len_y+1)
        self.im_array = np.array([[[0, 0, 0, 255] for i in range(int(abs(len_y)))] for j in range(int(abs(len_x)))])
        print(self.im_array.shape)

    def add_value_to_image(self, match):
        con = match.confidence
        # scale it manually to increase contrast
        if con > 60:
            contrast = 100
        elif con > 50:
            contrast = 40
        elif con > 35:
            contrast = 20 
        elif con > 20:
            contrast = 10
        else:
            contrast = 5 
        x_coord = int(self.x_scale_factor*(match.x- self.x_0))
        y_coord = int(self.y_scale_factor*(match.y-self.y_0))
        try:
            # just override previous value in image
            self.im_array[x_coord, y_coord] = [57, 255, 20, np.uint8(con*0.01*255)]
        except:
            pass
        x_to_add = int(self.x_scale_factor+2)
        y_to_add = int(self.y_scale_factor+2)
        if x_coord+ x_to_add -1 < len(self.im_array):
            if y_coord + y_to_add -1 < len(self.im_array[0]):
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


