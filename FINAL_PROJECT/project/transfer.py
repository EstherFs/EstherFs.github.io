import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

# Function to load an image from a file, and add a batch dimension.
def load_content_img(image_pixels):
    if image_pixels.shape[-1] == 4:
        image_pixels = Image.fromarray(image_pixels)
        img = image_pixels.convert('RGB')
        img = np.array(img)
        img = tf.convert_to_tensor(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 3:
        img = tf.convert_to_tensor(image_pixels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 1:
        raise Error('Grayscale images not supported! Please try with RGB or RGBA images.')
    print('Exception not thrown')

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image

# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck

# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()

  # Set model inputs.
  for index in range(len(input_details)):
    if input_details[index]["name"]=='Conv/BiasAdd':
      interpreter.set_tensor(input_details[index]["index"], style_bottleneck)
    elif input_details[index]["name"]=='content_image':
      interpreter.set_tensor(input_details[index]["index"], preprocessed_content_image)
  interpreter.invoke()

  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return stylized_image


style_image_path = "/home/ubuntu/project/style/maxresdefault.jpg"
#content_image_path = "/home/ubuntu/project/content/Golden_Gate_Bridge_2021.jpg"
style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3/int8/predict/1?lite-format=tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3/int8/transfer/1?lite-format=tflite')


async def transfer(content_image):
  #content_image = Image.open(content_image_path)
  content_image = content_image.convert('RGB')
  content_image.thumbnail((256, 256))

  # Convert the content image from Bytes to NumPy array.
  content_image = np.array(content_image)

  # Load the input images.
  content_image = load_content_img(content_image)
  style_image = load_img(style_image_path)

  # Preprocess the input images.
  preprocessed_content_image = preprocess_image(content_image, 384)
  preprocessed_style_image = preprocess_image(style_image, 256)




  content_blending_ratio = 0.1

  # Calculate style bottleneck for the preprocessed style image.
  style_bottleneck = run_style_predict(preprocessed_style_image)

  # Calculate style bottleneck of the content image.
  style_bottleneck_content = run_style_predict(
      preprocess_image(content_image, 256)
  )

  # Blend the style bottleneck of style image and content image
  style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
                             + (1 - content_blending_ratio) * style_bottleneck

  # Stylize the content image using the style bottleneck.
  stylized_image = run_style_transform(style_bottleneck_blended, preprocessed_content_image)


  im = Image.fromarray((stylized_image[0] * 255).astype(np.uint8)).convert('RGB')
  im.save("/home/ubuntu/project/tmp.jpeg")
  return im