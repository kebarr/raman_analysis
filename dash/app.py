# -*- coding: utf-8 -*-
#pip install dash==1.0.0  # The core dash backend
#pip install dash-daq==0.1.0 
# https://github.com/plotly/dash-recipes
import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from flask import Flask, flash, request
from werkzeug.utils import secure_filename
from convolution_matching import FindMaterial
from flask_caching import Cache

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
UPLOAD_FOLDER = 'data/'
THUMBNAIL_FOLDER = 'data/thumbnail/'


ALLOWED_EXTENSIONS = set(['txt', 'csv'])
IGNORED_FILES = set(['.gitignore'])

bootstrap = Bootstrap(app)


CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
#cache = Cache()
#cache.init_app(app.server, config=CACHE_CONFIG)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_file_name(filename):
    """
    If file was exist already, rename it and return a new name
    """

    i = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        name, extension = os.path.splitext(filename)
        filename = '%s_%s%s' % (name, str(i), extension)
        i += 1

    return filename


def create_thumbnail(image):
    try:
        base_width = 80
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image))
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
        img.save(os.path.join(app.config['THUMBNAIL_FOLDER'], image))

        return True

    except:
        print traceback.format_exc()
        return False


@app.route("/upload2", methods=['GET', 'POST'])
def upload2():
    print('upload 2')
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

                # create thumbnail after saving
                if mime_type.startswith('image'):
                    create_thumbnail(filename)
                
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mime_type, size=size)
            
            return simplejson.dumps({"files": [result.get_file()]})

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
    return send_from_directory(THUMBNAIL_FOLDER, filename=filename)


@app.route("/data/<string:filename>", methods=['GET'])
def get_file(filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER), filename=filename)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if not os.path.exists(os.path.join(app.config['assets_folder'], filename)):
            file.save(os.path.join(app.config['assets_folder'], filename))
    return '''
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
            '''
    

#@cache.memoize()
def add_data(df):
    find_material = FindMaterial(df)
    return find_material



#@cache.memoize()
def initialise_find_materials(material, subtract_baseline):
    find_material.initialise(material, subtract_baseline)



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