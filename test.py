from flask import Flask

app = Flask(__name__)

@app.before_first_request
def before_first():
    print("This runs before the first request.")

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
