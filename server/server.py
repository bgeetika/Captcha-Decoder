import json
import os
from bottle import route, run, static_file, request
config_file = open( 'config.json' )
config_data = json.load( config_file )
pth_xml     = config_data["paths"]["xml"]

@route('/recipes/')
def recipes_list():
    return { "success" : True }


run(host='localhost', port=8080, debug=True)

