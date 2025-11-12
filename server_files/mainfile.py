from flask import Flask, render_template, request, url_for, flash, redirect
import sqlite3
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'

#folder info
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 Mo
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



#utils functions used later
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def treatcv(post_id):
    return 2



#actual pages
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'cv' not in request.files:
            flash('Aucun fichier sélectionné.')
            return redirect(request.url)
        file = request.files['cv']
        if file.filename == '':
            flash('Aucun fichier choisi.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('Fichier reçu avec succès.')
            return redirect(url_for('upload_file'))
    return render_template('upload.html')


@app.route('/uploads', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'cv' not in request.files:
            flash('Aucun fichier sélectionné.')
            return redirect(request.url)
        file = request.files['cv']
        if file.filename == '':
            flash('Aucun fichier choisi.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('Fichier reçu avec succès.')
            return redirect(url_for('upload_files'))
    return render_template('uploads.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)










