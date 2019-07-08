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
from convolution_matching import FindMaterial
from flask_caching import Cache

import pandas as pd
import numpy as np
import scipy

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

find_material=FindMaterial()

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
    )
])

@cache.memoize()
def find_materials(function_name, kwargs):
    if function_name == 'add_data':
        find_material.add_data(kwargs)
    if function_name == 'find_matches':
        find_material.find_matches()
    return ['Succesfully initialised material finder']

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
        create_find_materials(df)
        children = ['Filename: ' + list_of_names[0] + 'uploaded']
        # would be nicer to hide then show items: https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
        # update entire options.... messy
        return [children, [{'value': 'with', 'label':'Find matches with baseline subtraction', 'disabled':False}, {'value':'without', 'label':'Find matches without baseline subtraction (baseline already subtracted)', 'disabled':False}]]
    return [children, [{'value': 'with', 'label':'Find matches with baseline subtraction', 'disabled':True}, {'value':'without', 'label':'Find matches without baseline subtraction (baseline already subtracted)', 'disabled':True}]]

# actually do the stuff...
@app.callback(
    [Output('clicked', 'children')],
    [Input('button', 'n_clicks')],
    [State('upload-options', 'value'),
    State('upload-data', 'filename')])
def upload_data(n_clicks, value, list_of_names):
    if n_clicks is not None:
        print('button pressed %d value: %s' % (n_clicks, str(value)))
        subtract_baseline = True if value=='with' else False
        initialise_find_materials('graphene_oxide', subtract_baseline)
        return ['Processed ' + str(value) + ' baseline sibtraction ']
    return ['No data to process yet']


if __name__ == '__main__':
    app.run_server(debug=True)