from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')
@app.route("/translate",  methods=['GET', 'POST'])
def translate():
    if request.method == "POST":
        dich = request.form['nom']
        print(dich)
    return render_template('translate.html', dich=dich)

if __name__ == "__main__":
    app.run(debug=True)
