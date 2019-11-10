from flask import Flask, render_template, request, redirect, flash, send_from_directory
from flask_bootstrap import Bootstrap
import os
from werkzeug.utils import secure_filename
import pandas as pd
from collections import defaultdict
from AutoML import *

UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE_MB = 1000
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

task = 'classification'
speed = 5
test_size = 0.2
target = None
metrics = None

global AutoML_Engine

def allowed_file(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(filename):

	ext = filename.rsplit('.', 1)[1].lower()

	if ext == 'csv':

		return pd.read_csv(filename)

	elif ext == 'txt':

		return pd.read_csv(filename)

	else:
		return pd.read_excel(filename)

def scale_speed(speed):

	if speed < 10:
		return 1
	else:
		return int(speed/10)

@app.route('/', methods=['GET', 'POST'])
def trainModel():
	global AutoML_Engine
	print('Running train model function')
	print(request)

	if request.method == 'POST' and 'file' in request.files and request.files['file']:
		file = request.files['file']

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(full_filename)
			#print('File upload success!')
			speed = scale_speed(int(request.form.get("speed")))
			task = request.form.get("task")
			target = request.form.get("targetColumn")

			AutoML_Engine = AutoMLEstimator(task = task, speed=speed, test_size=test_size)

			data = read_file(full_filename)
			AutoML_Engine.run_automl(data, target)
			metrics = AutoML_Engine.evaluate_model()

		return render_template('main_page.html', metrics=metrics)

	elif 'download' in request.form:

		AutoML_Engine.save_model(directory=app.config['UPLOAD_FOLDER'])
		filename = '{}.joblib'.format(AutoML_Engine.model_name)
		return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=filename, as_attachment=True)

	return render_template('main_page.html', metrics=None)


if __name__ == '__main__':
	app.run(debug=True)
