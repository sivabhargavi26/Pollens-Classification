from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('model/cnn_model.h5')

label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(128, 128))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_index = np.argmax(pred)
    class_name = label_encoder.inverse_transform([class_index])[0]

    return render_template('prediction.html', image=file.filename, label=class_name)

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)