import base64
import datetime
import io
from io import BytesIO
import os


from werkzeug.utils import secure_filename
from convolution_matching import FindMaterial
from flask_caching import Cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

import traceback

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from flask_bootstrap import Bootstrap

from lib.upload_file import uploadfile

# stting up docker... https://gist.github.com/PurpleBooth/635a35ed0cffb074014e https://runnable.com/docker/introduction-to-docker-compose
cwd = os.getcwd()
UPLOAD_FOLDER = cwd + '/uploads/'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SECRET_KEY = 'hard to guess string'
TEMPLATES_FOLDER = cwd + '/matching_templates/'
app.config['TEMPLATES_FOLDER'] = TEMPLATES_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'csv'])
IGNORED_FILES = set(['.gitignore'])

bootstrap = Bootstrap(app)


CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 60*60*3
}
cache = Cache()
cache.init_app(app, config=CACHE_CONFIG)



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_file_name(filename):
    """
    If file was exist already, rename it and return a new name
    """

    i = 1
    while os.path.exists(os.path.join('UPLOAD_FOLDER', filename)):
        name, extension = os.path.splitext(filename)
        filename = '%s_%s%s' % (name, str(i), extension)
        i += 1

    return filename


# add thing to not recompute if its one we've already done....

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    # clear the cache here to hopefully make sure it caches properly in subsequent fn calls
    cache.delete_memoized(find_material)
    if request.method == 'POST':
        files = request.files['file']

        if files:
            filename = secure_filename(files.filename)
            filename = gen_file_name(filename)
            mime_type = files.content_type

            if not allowed_file(files.filename):
                return render_template('file_uploaded.html', filename_not_uploaded=filename + " File type not allowed, not uploaded")

            else:
                # save file to disk
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'] , filename)
                if os.path.exists(uploaded_file_path):
                    # TODO if file exists, don't recompute, store peaks and reload to save computing time
                    return render_template('file_uploaded.html', filename=filename + " already uploaded, ready for use")
                files.save(uploaded_file_path)
                
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mime_type, size=size)
                return render_template('file_uploaded.html', filename=filename)




# think redis has a limit on size of command, as getting a socket timeout
# added 'maxmemory 10GB' to redis.conf # in top says 15M
@cache.memoize(timeout=60*60)
def find_material(filename, material, subtract_baseline):
    fm=  FindMaterial(filename, material, subtract_baseline)
    return fm


# https://gist.github.com/tebeka/5426211
def plot_random_baseline_example(fm, confidence="medium", number_to_plot=2):
    fig, ax = plt.subplots(1,1)
    fm.random_sample_compare_before_subtract_baseline.plot(ax=ax)
    fm.random_sample_compare_after_subtract_baseline.plot(ax=ax)
    ax.set_title("Example with/without baseline subtraction")
    ax.set(xlabel = 'Shift (cm$^{-1}$)')
    ax.set(ylabel='Intensity')
    io = BytesIO()
    fig.savefig(io, format='png')
    plt.close(fig)    
    io.seek(0)
    baseline_example = base64.encodestring(io.getvalue()).decode("utf-8")
    number_matches_med, data, match1, match2 = get_example_matches(fm, "medium", number_to_plot=2)
    number_matches_high = len(fm.matches.high_confidence)
    all_matches = len(fm.matches.matches)
    if number_matches_med == 0:
        return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches,  number_locations=fm.len, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=False) 
    template_data = fm.material.template
    fig, (ax1) = plt.subplots(1,1, sharex=True, sharey=True, figsize=(13, 5))
    ax1.set(xlabel = 'Shift (cm$^{-1}$)')
    ax1.set(ylabel='Intensity')
    ax1.plot(template_data)
    io = BytesIO()
    fig.savefig(io, format='png')
    plt.close(fig)
    io.seek(0)
    template = base64.encodestring(io.getvalue()).decode("utf-8")
    return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches,template = template, best_match = template, number_locations=fm.len, match_example = data, baseline_example=baseline_example, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=True, match1 = match1.to_dict(), match2=match2.to_dict())



def get_example_matches(fm, confidence="medium", number_to_plot=2):
    matches = fm.get_condifence_matches(confidence)
    number_matches = len(matches)
    if number_matches == 0:
        return 0, 0, None, None
    index_to_plot_1 = np.random.randint(0, number_matches)
    index_to_plot_2 = np.random.randint(0, number_matches)
    match1 = matches[index_to_plot_1]
    match2 = matches[index_to_plot_2]
    m1 = match1.spectrum
    m2 = match2.spectrum
    ymax = np.max([np.max(m1.values), np.max(m2.values)]) + 50
    #string = '%d matches found' % number_matches
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(13, 5))
    plt.ylim(ymin=-200, ymax=ymax)
    ax1.set(xlabel = 'Shift (cm$^{-1}$)')
    ax1.set(ylabel='Intensity')
    ax2.set(xlabel = 'Shift (cm$^{-1}$)')
    ax2.set(ylabel='Intensity')
    m1.plot(ax=ax1)
    m2.plot(ax=ax2)
    io = BytesIO()
    fig.savefig(io, format='png')
    io.seek(0)
    img_str = base64.encodestring(io.getvalue()).decode("utf-8")
    plt.close(fig)
    return number_matches, img_str, match1, match2

def plot_example_match(fm):
    number_matches_med, data, match1, match2 = get_example_matches(fm, "medium", number_to_plot=2)
    number_matches_high = len(fm.matches.high_confidence)
    all_matches = len(fm.matches.matches)
    template_data = fm.material.template
    fig, (ax1) = plt.subplots(1,1, sharex=True, sharey=True, figsize=(13, 5))
    ax1.set(xlabel = 'Shift (cm$^{-1}$)')
    ax1.set(ylabel='Intensity')
    ax1.plot(template_data)
    io = BytesIO()
    fig.savefig(io, format='png')
    plt.close(fig)
    io.seek(0)
    template = base64.encodestring(io.getvalue()).decode("utf-8")
    if number_matches_med == 0:
        return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches, template = template,  number_locations=fm.len, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=False) 
    return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches, template = template,  number_locations=fm.len, match_example=data, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=False, match1 = match1.to_dict(), match2=match2.to_dict())

@app.route('/uploadajax', methods = ['POST'])
def upload_image():
    if request.method == 'POST':
        files = request.files['file']
        if files:
            filename = secure_filename(files.filename)
            filename = gen_file_name(filename)
            if filename.endswith(".bmp") or filename.endswith(".jpeg"):
                mime_type = files.content_type
                # save file to disk
                uploaded_file_path = os.path.join(UPLOAD_FOLDER, filename)
                files.save(uploaded_file_path)
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)
                uploadfile(name=filename, type=mime_type, size=size)
                data_filename = request.form.get("filename")
                material = request.form.get("material")
                sb = request.form.get("sb")
                sb_bool = True if sb == "True" else False
                output_filename = request.form.get("output_filename")
                fm = find_material(data_filename, material, sb_bool)
                fm.overlay_match_positions(uploaded_file_path, output_filename)
                with open(output_filename, 'rb') as image:
                    img_str = base64.b64encode(image.read())
                return {'image': img_str, 'output_filename': output_filename}



# now need to wire this in to plot example baseline
@app.route('/plot_med', methods = ['POST'])
def plot_med():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    fm = find_material(data_filename, material, sb_bool)
    _, img_str, match1, match2 = get_example_matches(fm, confidence="medium", number_to_plot=2)
    # actually_do_the_stuff is called.... whyu!?!?!?!?!?
    return {'image': img_str, 'match1_confidence': match1.confidence, 'match2_confidence': match2.confidence, \
        'd_intensity1':match1.peak_data[0][1], 'g_intensity1': match1.peak_data[1][1], 'peak_ratio1': match1.peak_ratio, \
            'd_intensity2':match2.peak_data[0][1], 'g_intensity2': match2.peak_data[1][1], 'peak_ratio2': match2.peak_ratio,\
                'x1': match1.x, 'y1': match1.y, 'x2': match2.x, 'y2': match2.y}

@app.route('/plot_high', methods = ['POST'])
def plot_high():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    fm = find_material(data_filename, material, sb_bool)
    _, img_str, match1, match2 = get_example_matches(fm, confidence="high", number_to_plot=2)
    return {'image': img_str, 'match1_confidence': match1.confidence, 'match2_confidence': match2.confidence, \
        'd_intensity1':match1.peak_data[0][1], 'g_intensity1': match1.peak_data[1][1], 'peak_ratio1': match1.peak_ratio, \
            'd_intensity2':match2.peak_data[0][1], 'g_intensity2': match2.peak_data[1][1], 'peak_ratio2': match2.peak_ratio,\
                'x1': match1.x, 'y1': match1.y, 'x2': match2.x, 'y2': match2.y}

@app.route('/download_high', methods = ['POST'])
def download_high():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    fm = find_material(data_filename, material, sb_bool)
    # output two files: one with a different spectrum on each row
    # going to be a faff, just do csv with spectrum at end
    with open('high_conf.csv', 'w') as f:
        header = 'x, y, d intensity, g intensity, d/g, confidence, shifts \n'
        f.write(header)
        for match in fm.matches.matches:
            spectrum_string = ''
            for i in match.spectrum:
                spectrum_string += str(i) + ', '
            line = str(match.x) + ', ' + str(match.y) + ', ' + str(match.peak_data[0][1]) + ', ' + str(match.peak_data[1][1]) + ', ' + str(match.confidence) + ', ' + spectrum_string + '\n'
            f.write(line)
        wavelengths = ', , , , , '
        for w in fm.wavelengths:
            wavelengths += ',' + str(w) 
        f.write(wavelengths)
    return send_file('high_conf.csv', as_attachment=True)

@app.route('/download_med', methods = ['POST'])
def download_med():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    fm = find_material(data_filename, material, sb_bool)
    # output two files: one with a different spectrum on each row
    # going to be a faff, just do csv with spectrum at end
    with open('med_conf.csv', 'w') as f:
        header = 'x, y, d intensity, g intensity, d/g, confidence, shifts \n'
        f.write(header)
        for match in fm.matches.matches:
            spectrum_string = ''
            for i in match.spectrum:
                spectrum_string += str(i) + ', '
            line = str(match.x) + ', ' + str(match.y) + ', ' + str(match.peak_data[0][1]) + ', ' + str(match.peak_data[1][1]) + ', ' + str(match.peak_ratio)+ ', ' + str(match.confidence) + ', ' + spectrum_string + '\n'
            f.write(line)
        wavelengths = ', , , , '
        for w in fm.wavelengths:
            wavelengths += ',' + str(w) 
        f.write(wavelengths)
    return send_file('med_conf.csv', as_attachment=True)


@app.route('/download_image', methods=['GET', 'POST'])
def download_image():
    filename = request.form['output_filename']
    uploads = os.path.join(app.root_path)
    return send_from_directory(directory=uploads, filename=filename, as_attachment=True)

# TODO: export sample spectra, shuffle random spectrum in case poor match shown
# show what i've compared and highest confidence match. fix cache.

@app.route('/find_peaks', methods=['POST'])
def actually_do_the_stuff():
    option = request.form['baseline']
    filename = UPLOAD_FOLDER + request.form['filename']
    subtract_baseline = True if option == 'with' else False
    fm = find_material(filename, u'graphene_oxide', subtract_baseline)
    if subtract_baseline == True:
        return plot_random_baseline_example(fm)
    return plot_example_match(fm)

# @app.route('download-image', methods=['POST'])
# def send_image_to_user():
#     filename = 
#     send_from_directory()

@app.route('/')
def home():
    return render_template('file_uploaded.html')

@app.route('/test', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


# minimum to do!!
# id/ig ratio - find max in g peak region and max in d peak region, not robust to cosmic rays but my baselining removes these.
# show for shown match, confidence score, d/g ratio
# user output - image of spectrum, of currently shown match (so need to separate somehow, as its bytecode!!) 