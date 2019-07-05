from app import app
from flask import session, Response
from convolution_matching import FindMaterial
from shelljob import proc

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/adshjhkfhjfhh\xec\n\xb5\xd0\xba'

PATH_FOLDER='/'

#@app.route('/')
#@app.route('/index')
#def index():
#    return "Hello, World!"


finder_dict = {}

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

# usage: http://127.0.0.1:5000/initialise_matcher/graphene_oxide/Users/mbbxhkb2/Documents/raman_imaging/raw_baselined_comp/pos_baselined.csv/0
@app.route('/initialise_matcher/<material_name>/<path:data_filename>/<int:subtract_baseline>')
def initialise_matcher(material_name, data_filename, subtract_baseline=False):
    finder = FindMaterial(material_name, PATH_FOLDER+data_filename, bool(subtract_baseline))
    finder_dict['finder'] = finder 
    return "material_name: %s, data_filename: %s, subtract_baseline: %d " % (material_name, data_filename, subtract_baseline)

@app.route('/find_matches')
def find_matches():
    if 'finder' in finder_dict:
        finder_dict['finder'].find_matches()
        return "Success"
    return "Please upload data"


@app.route( '/stream' )
def stream():
    g = proc.Group()
    p = g.run( [ "bash", "-c", "for ((i=0;i<100;i=i+1)); do echo $i; sleep 1; done" ] )

    def read_process():
        while g.is_pending():
            lines = g.readlines()
            for proc, line in lines:
                yield line

    return Response( read_process(), mimetype= 'text/plain' )