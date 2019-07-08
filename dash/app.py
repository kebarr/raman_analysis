# -*- coding: utf-8 -*-
#pip install dash==1.0.0  # The core dash backend
#pip install dash-daq==0.1.0 
# https://github.com/plotly/dash-recipes
import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from convolution_matching import FindMaterial


import pandas as pd
import numpy as np
import scipy

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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



#def parse_contents(contents, filename, date):
#    content_type, content_string = contents.split(',')

#    decoded = base64.b64decode(content_string)
#    try:
#        if 'csv' or 'txt' in filename:
            # Assume that the user uploaded a CSV file
#            df = pd.read_csv(
#                io.StringIO(decoded.decode('utf-8')))
#        else:
#            raise ValueError("please upload .csv or .txt file")
#    except Exception as e:
#        print(e)
#        return html.Div([
#            'There was an error processing this file.'
#        ])

#    return html.A("Succesfully uploaded file")

# try to chain callbacks so it shows filename before starting to update
#def fun1(input1):
#      ...
#     callbackfun(input)

#def callbackfun(input):
#      ...
#     return data

#app.callback(Output('output', 'children'))(callbackfun)
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
    #    children = [
    #        parse_contents(c, n, d) for c, n, d in
    #        zip(list_of_contents, list_of_names, list_of_dates)]
        children = ['Filename: ' + list_of_names[0]]
    # would be nicer to hide then show items: https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
    # update entire options.... messy
    return [children, [{'value': 'with', 'label':'Upload with baseline subtraction', 'disabled':False}, {'value':'without', 'label':'Upload without baseline subtraction', 'disabled':False}]]

# actually do the stuff...
@app.callback(
    [Output('clicked', 'children')],
    [Input('button', 'n_clicks')],
    [State('upload-options', 'value')])
def upload_data(n_clicks, value):
    if n_clicks is not None:
        print('button pressed %d' % (n_clicks))
        return ['Processing ' + str(value) + ' baseline sibtraction ']
    return ['No data to process yet']


if __name__ == '__main__':
    app.run_server(debug=True)