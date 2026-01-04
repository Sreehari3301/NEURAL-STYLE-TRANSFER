import os
import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time

def tensor_to_image(tensor):
    """Converts a tensor to a PIL image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    """Loads an image and limits its maximum dimension to 512 pixels."""
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Calculates the Gram matrix for style representation."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    """Helper to keep pixel values between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    # Style Loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)

    # Content Loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)
    
    return style_loss + content_loss

def run_style_transfer(content_path, style_path, iterations=1000, content_weight=1e4, style_weight=1e-2):
    # Load images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Define content and style layers
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Initialize model
    extractor = StyleContentModel(style_layers, content_layers)

    # Precalculate target values
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # The image we are optimizing
    image = tf.Variable(content_image)

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets, 
                                    style_weight, content_weight, style_layers, content_layers)
            # Add Total Variation Loss to reduce high-frequency noise
            loss += tf.image.total_variation(image) * 30 

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    print("Starting optimization...")
    start_time = time.time()

    for i in range(iterations):
        train_step(image)
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations}")
            # Save intermediate result
            temp_img = tensor_to_image(image)
            temp_img.save(f'stylized_iteration_{i}.png')

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f}s")
    
    return image

if __name__ == "__main__":
    CONTENT_PATH = 'content.png'
    STYLE_PATH = 'style.png'
    OUTPUT_PATH = 'stylized_image.png'

    if not os.path.exists(CONTENT_PATH) or not os.path.exists(STYLE_PATH):
        print(f"Error: Make sure '{CONTENT_PATH}' and '{STYLE_PATH}' exist.")
    else:
        # Run transfer
        result = run_style_transfer(CONTENT_PATH, STYLE_PATH, iterations=200)
        
        # Save result
        final_img = tensor_to_image(result)
        final_img.save(OUTPUT_PATH)
        print(f"Success! Result saved to {OUTPUT_PATH}")
