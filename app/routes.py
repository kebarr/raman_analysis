from app import app

#@app.route('/')
#@app.route('/index')
#def index():
#    return "Hello, World!"

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

@app.route('/initialise_matcher/<material_name>/<data_filename>/<int:subtract_baseline>')
def initialise_matcher(material_name, data_filename, subtract_baseline):
    return "material_name: %s, data_filename: %s, subtract_baseline: %d " % (material_name, data_filename, subtract_baseline)



