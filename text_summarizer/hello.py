from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/test")
def test():
    return "Hello World! This is a test"

@app.route("/textSum")
def textSum():
    return "This is a text Summarizer Service"

@app.route("/user/<string:name>/")
def getMember(name):
    return "Hi This is %s!!" % name

if __name__ == "__main__":
    app.run()
