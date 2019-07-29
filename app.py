import base64
import datetime
import io
import os
from cStringIO import StringIO


from flask import Flask, flash, request
from werkzeug.utils import secure_filename
from convolution_matching import FindMaterial
from flask_caching import Cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

import simplejson
import traceback

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap

from lib.upload_file import uploadfile

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
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
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
                print(result)
                return render_template('file_uploaded.html', filename=filename)




# think redis has a limit on size of command, as getting a socket timeout
# added 'maxmemory 10GB' to redis.conf # in top says 15M
@cache.memoize(timeout=60*60)
def find_material(filename, material, subtract_baseline):
    print(filename, material, subtract_baseline)
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
    io = StringIO()
    fig.savefig(io, format='png')
    plt.close(fig)
    baseline_example = base64.encodestring(io.getvalue())
    number_matches_med, data = get_example_matches(fm, "medium", number_to_plot=2)
    number_matches_high = len(fm.matches.high_confidence)
    all_matches = len(fm.matches.matches)
    template_name = os.path.join(app.config['TEMPLATES_FOLDER'] , fm.material.name)
    print(template_name)
    with open(template_name, 'rb') as image:
        template = base64.b64encode(image.read())
    return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches,template = template, best_match = template, number_locations=fm.len, match_example = data, baseline_example=baseline_example, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=True)



def get_example_matches(fm, confidence="medium", number_to_plot=2):
    print(confidence)
    matches = fm.get_condifence_matches(confidence)
    number_matches = len(matches)
    print(len(matches[0]), len(matches))
    index_to_plot_1 = np.random.randint(0, number_matches)
    index_to_plot_2 = np.random.randint(0, number_matches)
    m1 = matches[index_to_plot_1][2]
    m2 = matches[index_to_plot_2][2]
    cp1 = matches[index_to_plot_1][3]
    cp2 = matches[index_to_plot_2][3]
    c1 = matches[index_to_plot_1][1]
    c2 = matches[index_to_plot_2][1]
    print("confidence 1: %d, confidence 2: %d" % (c1, c2) )
    print(cp1)
    print(cp2)
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
    io = StringIO()
    fig.savefig(io, format='png')
    plt.close(fig)
    return number_matches, base64.encodestring(io.getvalue())

def plot_example_match(fm):
    number_matches_med, data = get_example_matches(fm, "medium", number_to_plot=2)
    number_matches_high = len(fm.matches.high_confidence)
    all_matches = len(fm.matches.matches)
    template_data = fm.material.template
    fig, (ax1) = plt.subplots(1,1, sharex=True, sharey=True, figsize=(13, 5))
    ax1.set(xlabel = 'Shift (cm$^{-1}$)')
    ax1.set(ylabel='Intensity')
    ax1.plot(template_data)
    io = StringIO()
    fig.savefig(io, format='png')
    plt.close(fig)
    template = base64.encodestring(io.getvalue())
    if number_matches_med == 0:
        return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches, template = template,  number_locations=fm.len, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=False) 
    return render_template('plot_data.html', number_matches_med=number_matches_med, number_matches_high=number_matches_high, all_matches= all_matches, template = template,  number_locations=fm.len, match_example=data, filename=fm.data_filename, material="graphene_oxide", subtract_baseline=False)

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


# (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', u'graphene_oxide', False)
# (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', u'graphene_oxide', False)
# now need to wire this in to plot example baseline
@app.route('/plot_med', methods = ['POST'])
def plot_med():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    print(data_filename, material, sb)
    fm = find_material(data_filename, material, sb_bool)
    print("found material")
    _, img_str = get_example_matches(fm, confidence="medium", number_to_plot=2)
    print("got example matches")
    # actually_do_the_stuff is called.... whyu!?!?!?!?!?
    return {'image': img_str}

@app.route('/plot_high', methods = ['POST'])
def plot_high():
    data_filename = request.form.get("filename")
    material = request.form.get("material")
    sb = request.form.get("sb")
    sb_bool = True if sb == "True" else False
    fm = find_material(data_filename, material, sb_bool)
    print("found material")
    _, img_str = get_example_matches(fm, confidence="high", number_to_plot=2)
    print("got example matches")
    return {'image': img_str}


@app.route('/download_image', methods=['GET', 'POST'])
def download_image():
    filename = request.form['output_filename']
    uploads = os.path.join(app.root_path)
    return send_from_directory(directory=uploads, filename=filename, as_attachment=True)

# TODO: export sample spectra, shuffle random spectrum in case poor match shown
# show what i've compared and highest confidence match. fix cache.

@app.route('/find_peaks', methods=['POST'])
def actually_do_the_stuff():
    print("called")
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


# grep -rno 'jquery-file-upload.appspot.com' .
#./index.html:15:jquery-file-upload.appspot.com
#./templates/index.html:15:jquery-file-upload.appspot.com
#./flask-file-uploader/static/js/main.js:37:jquery-file-upload.appspot.com
#./flask-file-uploader/static/js/main.js:49:jquery-file-upload.appspot.com
#./flask-file-uploader/static/js/app.js:19:jquery-file-upload.appspot.com
#./flask-file-uploader/templates/index.html:15:jquery-file-upload.appspot.com

#sed -i -e 's,jquery-file-upload.appspot.com,flask-file-uploader/static/js/jQuery-File-Upload-9.32.0/,g' ./index.html
#sed -i -e 's,jquery-file-upload.appspot.com,flask-file-uploader/static/js/jQuery-File-Upload-9.32.0/,g' ./templates/index.html
#sed -i -e 's,jquery-file-upload.appspot.com,flask-file-uploader/static/js/jQuery-File-Upload-9.32.0/,g' ./flask-file-uploader/static/js/main.js
#sed -i -e 's,jquery-file-upload.appspot.com,flask-file-uploader/static/js/jQuery-File-Upload-9.32.0/,g' ./flask-file-uploader/static/js/app.js
#sed -i -e 's,jquery-file-upload.appspot.com,flask-file-uploader/static/js/jQuery-File-Upload-9.32.0/,g' ./flask-file-uploader/templates/index.html
#sed -i -e 's,http://127.0.0.1:5000/flask-file-uploader,flask-file-uploader,g' ./flask-file-uploader/static/js/main.js
#sed -i -e 's,http://127.0.0.1:5000/flask-file-uploader,flask-file-uploader,g' ./flask-file-uploader/static/js/app.js
#sed -i -e 's,http://127.0.0.1:5000/flask-file-uploader,flask-file-uploader,g' ./index.html
#sed -i -e 's,http://127.0.0.1:5000/flask-file-uploader,flask-file-uploader,g'./templates/index.html
#sed -i -e 's,http://127.0.0.1:5000/flask-file-uploader,flask-file-uploader,g' ./flask-file-uploader/templates/index.html

#sed -i -e 's,127.0.0.1:5000/flask-file-uploader,jquery-file-upload.appspot.com,g' ./flask-file-uploader/static/js/main.js
#sed -i -e 's,127.0.0.1:5000/flask-file-uploader,jquery-file-upload.appspot.com,g' 
# (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', 'graphene_oxide', False)

# first call: (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', 'graphene_oxide', u'False')
# is now: (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', u'graphene_oxide', u'False')
#         (u'uploads/D_M1_Spleen_Slide3_2019_02_11_14_43_47.txt', u'graphene_oxide', u'False')
# theyt're exactly the same so pwhat is the issue!!!
# id/ig ratio