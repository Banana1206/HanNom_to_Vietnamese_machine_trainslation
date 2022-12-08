from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')


@app.route("/login")
def login():
    return render_template('login.html')

if __name__ == "__main__":
    app.run(debug=True)
