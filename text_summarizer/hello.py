import sys
sys.path.append('/home/atul/Git/DeepLearning/text_summarizer/backEnd')
import config
from  Summarizer import *
from spacy.lang.en import STOP_WORDS
import en_core_web_md
from gensim.models import Word2Vec as wv
import json
import time
import pathlib


from flask import Flask, abort, request, jsonify, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
# @app.route("/textSum", methods=['GET', 'POST'])
# def hello():
#     print("Chal")
#     #content = request.get_json(silent=True)
#     #data = flask.request.json
#     #print(content)
#     #filename = json.loads(upload_data)['doc_name'];
#     return 'content'


@app.route('/textSum1', methods=['POST','GET'])
#@cross_origin()
def foo():
    #resp = request.get_json(silent=True)
    # resp = list(request.args.keys())[1]
    # data = json.loads(resp)#['doc_name']
    # js = dict(data[0])
    # filename = js['doc_name']
    # if not filename:
    #     print("file is not there")
    # else:
    #     print(filename)
    #     #rint(data)

    docs = {}
    docs['doc_type'] = "text"
    docs['doc_summary'] = str(time.time())
    summary_json = json.dumps(docs)
    response = Response(summary_json, status=200, mimetype='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')

    print(response)
    print(docs)
    return response


@app.route("/textSum", methods=['POST','GET'])
def extSummary():
    tick = time.time()
    print("tick - "+ str(tick))
    dir_path='/home/atul/Downloads/'
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    resp = list(request.args.keys())[1]
    data = json.loads(resp)#['doc_name']
    js = dict(data[0])
    filename = js['doc_name']
    if not filename:
        print("file is not there")
        sys.exit()
    else:
        filename = dir_path + filename
        print(filename)

    # Initializing the no of sentences in extracted summary
    NUM_OF_SENTENCES = config.NUM_OF_SENTENCES

    # Extracting text from input file
    print("Step 1: Data Parsing")
    #parsed_text = pdfparser("/home/atul/Git/DeepLearning/text_summarizer/backEnd/learnings.pdf")
    file_ext = filename[-4:]
    if file_ext=='.pdf':
        doc_type='pdf'
        print("File type is pdf")
        parsed_text = pdfparser(filepath)
    elif file_ext=='.txt':
        doc_type='text'
        print("File type is text")
        parsed_text = open(filename).read()
    else:
        print("File format is not correct")
        sys.exit()

    final_text = "".join(parsed_text).replace('\n',' ')
    text = re.sub('[^A-Za-z0-9%,.$&() ]','',final_text)

    print("Step 2: English Language Model loading")
    # Loading the Spacy English model
    # nlp = en_core_web_md.load()


    stopwords = {}
    for word in STOP_WORDS:
        stopwords[word]=''

    print("Step 3: Word Embedding loading")
    # Loading the Word Embeddings
    # model = wv.load('/home/atul/Git/wiki_model_new/wiki.en.text.model')

    print("Step 4: Extracting Summary")
    # summ = Summarizer(model, nlp)
    # summary = summ.get_summary(text,NUM_OF_SENTENCES)
    #
    # summary_json = {}
    # summary_json['doc_type'] = doc_type
    # summary_json['doc_summary'] = summary

    docs = {}
    docs['doc_type'] = "text"
    docs['doc_summary'] = str(time.time())
    summary_json = json.dumps(docs)

    result = json.dumps(summary_json)
    response = Response(result, status=200, mimetype='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')

    print(response)
    print(result)
    tock = time.time()
    print("Total Time %s"% str(tock-tick))
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
