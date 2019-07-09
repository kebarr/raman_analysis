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

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=UPLOAD_FOLDER)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print UPLOAD_FOLDER
SECRET_KEY = 'hard to guess string'
UPLOAD_FOLDER = 'data/'
THUMBNAIL_FOLDER = 'data/thumbnail/'
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg', 'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])
IGNORED_FILES = set(['.gitignore'])


CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Div([
    dcc.RadioItems(
        id='upload-options',
        options=[{'value': 'with', 'label':'Upload with baseline subtraction', 'disabled':True}, {'value':'without', 'label':'Upload without baseline subtraction', 'disabled':True}]
    ),
    
    ]),
    html.Div([
    # can't make hide/disable button work.....
    html.Button('Upload', id='button', disabled=True)]
    ),
    html.Div([html.Div('Ready to upload file', id='clicked')]),
     #style={'display' : 'none'}),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    ), html.Div( 
        children=[
            html.Iframe(id='iframe-upload',src='/upload'),
            html.Div(id='output')
                ]
)
])

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


@app.server.route("/upload2", methods=['GET', 'POST'])
def upload2():
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


@app.server.route("/delete/<string:filename>", methods=['DELETE'])
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
@app.server.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
    return send_from_directory(THUMBNAIL_FOLDER, filename=filename)


@app.server.route("/data/<string:filename>", methods=['GET'])
def get_file(filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER), filename=filename)


@app.server.route('/upload', methods=['GET', 'POST'])
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
    

@cache.memoize()
def add_data(df):
    find_material = FindMaterial(df)
    return find_material



@cache.memoize()
def initialise_find_materials(material, subtract_baseline):
    find_material.initialise(material, subtract_baseline)

@app.callback(
    [Output('button', 'disabled'),
    Output('clicked', component_property='style')],
    [Input('upload-options', 'value')])
def set_baseline_option(value):
    if value is not None:
        print 'selected ' + str(value)
        return [False, {'display' : 'block'}]
    return [False, {'display' : 'block'}]

# proposed logic: select file to upload, display name. click button to say 'process' - this will parse data and 

# try to edit this so that it doesn't download contents yet- move input to button callback
@app.callback([Output('output-data-upload', 'children'),# this is what is returned
                Output('upload-options', 'options')],
              [Input('upload-data', 'contents')], # input
              [State('upload-data', 'filename')]) 
def update_output(list_of_contents, list_of_names):
    children = []
    if list_of_contents is not None:
        content_type, content_string = list_of_contents[0].split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')))
        fm = add_data(df)
        children = ['Filename: ' + list_of_names[0] + 'uploaded']
        # would be nicer to hide then show items: https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
        # update entire options.... messy
        return [children, [{'value': 'with', 'label':'Find matches with baseline subtraction', 'disabled':False}, {'value':'without', 'label':'Find matches without baseline subtraction (baseline already subtracted)', 'disabled':False}]]
    return [children, [{'value': 'with', 'label':'Find matches with baseline subtraction', 'disabled':True}, {'value':'without', 'label':'Find matches without baseline subtraction (baseline already subtracted)', 'disabled':True}]]

# actually do the stuff...
# button callback
@app.callback(
    [Output('clicked', 'children')],
    [Input('button', 'n_clicks')],
    [State('upload-options', 'value'),
    State('upload-data', 'filename')])
def upload_data(n_clicks, value, list_of_names):
    if n_clicks is not None:
        print('button pressed %d value: %s' % (n_clicks, str(value)))
        subtract_baseline = True if value=='with' else False
        # surely the entire point of memoization is so that its run once, so it should be returning the result from last time, when there was actual input?
        # its being computed with a different value to last time though, which is why its getting overwritten.
        # so need a function we can call to get up to date find_materials, without 
        fm = add_data()
        fm.find_matches('graphene_oxide', subtract_baseline)
        return ['Processed ' + str(value) + ' baseline sibtraction ']
    return ['No data to process yet']



if __name__ == '__main__':
    app.run_server(debug=True)