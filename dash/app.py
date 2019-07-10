# -*- coding: utf-8 -*-
#pip install dash==1.0.0  # The core dash backend
#pip install dash-daq==0.1.0 
# https://github.com/plotly/dash-recipes
import base64
import datetime
import io
import os
from cStringIO import StringIO


from flask import Flask, flash, request
from werkzeug.utils import secure_filename
from convolution_matching import FindMaterial
from flask_caching import Cache

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

import PIL
from PIL import Image
import simplejson
import traceback

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename

from lib.upload_file import uploadfile

# worked and now doesn't!!
  #571  pip install Flask
  #572  export FLASK_APP=microblog.py
  #573  flask run
 # 574  export FLASK_APP=find_peaks.py
#flask run

# now try this: https://community.plot.ly/t/dash-upload-component-decoding-large-files/8033/11

cwd = os.getcwd()
UPLOAD_FOLDER = cwd + '/uploads'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print UPLOAD_FOLDER
SECRET_KEY = 'hard to guess string'
UPLOAD_FOLDER = 'uploads/'
THUMBNAIL_FOLDER = 'data/thumbnail/'


ALLOWED_EXTENSIONS = set(['txt', 'csv'])
IGNORED_FILES = set(['.gitignore'])

bootstrap = Bootstrap(app)


CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
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
    print('upload')
    if request.method == 'POST':
        files = request.files['file']

        if files:
            filename = secure_filename(files.filename)
            filename = gen_file_name(filename)
            mime_type = files.content_type

            if not allowed_file(files.filename):
                result = uploadfile(name=filename, type=mime_type, size=0, not_allowed_msg="File type not allowed")

            else:
                # save file to disk
                uploaded_file_path = os.path.join(UPLOAD_FOLDER, filename)
                files.save(uploaded_file_path)
                
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mime_type, size=size)
                print result.get_file()
            
            return render_template('file_uploaded.html', filename=filename)

    if request.method == 'GET':
        # get all file in ./data directory
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER,f)) and f not in IGNORED_FILES ]
        
        file_display = []

        for f in files:
            size = os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
            file_saved = uploadfile(name=f, size=size)
            file_display.append(file_saved.get_file())

        return simplejson.dumps({"files": file_display})

    return redirect(url_for('index'))


@app.route("/delete/<string:filename>", methods=['DELETE'])
def delete(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)

            if os.path.exists(file_thumb_path):
                os.remove(file_thumb_path)
            
            return simplejson.dumps({filename: 'True'})
        except:
            return simplejson.dumps({filename: 'False'})


# serve static files
@app.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename=filename)
    
# serve static files
@app.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
    return send_from_directory(THUMBNAIL_FOLDER, filename=filename)


@app.route("/data/<string:filename>", methods=['GET'])
def get_file(filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER), filename=filename)


    


@cache.memoize()
def find_material(filename, material, subtract_baseline):
    return FindMaterial(filename, material, subtract_baseline)

@cache.memoize()
def find_matches(fm):
    fm.find_matches()
    return fm

def start_find_materials():
    return render_template('file_uploaded.html', status='Loading data into material finder')

def find_materials_initialised():
    return render_template('file_uploaded.html', status='Data loaded into material finder, starting to look for matches')


# https://gist.github.com/tebeka/5426211
def plot_random_baseline_example(fm):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fm.random_sample_compare_before_subtract_baseline.plot(ax=ax)
    fm.random_sample_compare_after_subtract_baseline.plot(ax=ax)
    ax.set_title("Example with/without baseline subtraction")
    io = StringIO()
    fig.savefig(io, format='png')
    data = base64.encodestring(io.getvalue())
    return render_template('plot_data.html', number_matches=len(fm.matches), number_locations=len(fm.data), baseline_example=data)



# actually make a new template for this, with graphs!!
def plot_example_match(fm):
    number_matches = len(fm.matches.matches)
    index_to_plot_1 = np.random.randint(0, number_matches)
    index_to_plot_2 = np.random.randint(0, number_matches)
    m1 = fm.data.iloc[index_to_plot_1]
    m2 = fm.data.iloc[index_to_plot_2]
    ymax = np.max([np.max(m1.values), np.max(m2.values)]) + 50
    string = '%d matches found' % number_matches
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(13, 5))
    plt.ylim(ymin=-200, ymax=ymax)
    m1.plot(ax=ax1)
    m2.plot(ax=ax2)
    io = StringIO()
    fig.savefig(io, format='png')
    data = base64.encodestring(io.getvalue())
    return render_template('plot_data.html', number_matches=number_matches, number_locations=len(fm.data), match_example=data)

@app.route('/plot_match_positions_on_image', methods=['GET', 'POST'])
def plot_match_positions_on_image():
    pass

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    filename = request.args.get('filename')
    filename = gen_file_name(filename)

    # save file to disk
    uploaded_file_path = os.path.join(UPLOAD_FOLDER, filename)
    files.save(uploaded_file_path)

    # get file size after saving
    size = os.path.getsize(uploaded_file_path)

    # return json for js call back
    result = uploadfile(name=filename, type=mime_type, size=size)
    return simplejson.dumps({"files": [result.get_file()]})


#@app.route('show_image/<filename>')
#def show_image(filename):
#    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/find_peaks', methods=['POST'])
def actually_do_the_stuff():
    print('called find peaks', request.form.keys)
    option = request.form['baseline']
    filename = UPLOAD_FOLDER + request.form['filename']
    print option, filename
    subtract_baseline = True if option == 'with' else False
    start_find_materials()
    fm = find_material(filename, 'graphene_oxide', subtract_baseline)
    find_materials_initialised()
    fm = find_matches(fm)
    if subtract_baseline:
        plot_random_baseline_example(fm)
    return plot_example_match(fm)

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