import os
import time
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from nst_vgg19 import load_img, tensor_to_image, StyleContentModel, style_content_loss, clip_0_1

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model once at startup to save time
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
extractor = StyleContentModel(style_layers, content_layers)

def run_nst_logic(content_path, style_path, output_path, iterations=100):
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets, 
                                    1e-2, 1e4, style_layers, content_layers)
            loss += tf.image.total_variation(image) * 30 
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    for i in range(iterations):
        train_step(image)
    
    final_img = tensor_to_image(image)
    final_img.save(output_path)
    return output_path

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    content_file = request.files['content']
    style_file = request.files['style']
    
    if not (allowed_file(content_file.filename) and allowed_file(style_file.filename)):
        return jsonify({'error': 'Invalid file format. Only PNG, JPG, and JPEG are supported.'}), 400
    
    content_name = secure_filename(content_file.filename)
    style_name = secure_filename(style_file.filename)
    
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"c_{int(time.time())}_{content_name}")
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], f"s_{int(time.time())}_{style_name}")
    
    content_file.save(content_path)
    style_file.save(style_path)
    
    result_filename = f"out_{int(time.time())}.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    # Run NST (limited iterations for web response)
    run_nst_logic(content_path, style_path, result_path, iterations=50)
    
    return jsonify({'result_url': f'/results/{result_filename}'})

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
