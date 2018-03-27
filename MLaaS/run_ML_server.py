from sklearn.externals import joblib
from flask import Flask,  request,
from flask_cors import CORS, cross_origin
import numpy as np
import re

app = Flask(__name__)
CORS(app)

@app.route('/iris', methods=['POST','GET'])
@cross_origin()
def predict_species():
    # Request
    # curl -d 'param="4.9,1,4,  0.5"' -X POST "http://0.0.0.0:5000/iris"
    clf = joblib.load('iris.model')
    req = re.sub('"', '', request.values['param'])
    q = np.array(req.split(','), dtype=np.float32).reshape(1, -1)
    #print(q)

    color_code = dict({0: 'Iris-Setosa', 1: 'Iris-Versicolour', 2: 'Iris-Virginica'})
    colors_pred = str([color_code[x] for x in clf.predict(q)])
    return colors_pred


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
