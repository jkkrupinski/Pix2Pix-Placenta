import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from matplotlib import pyplot as plt

# The facade training set consist of 400 images
BUFFER_SIZE = 2060
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

PATH = 'FetReg/'

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def resize_single(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def normalize_single(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image

def load(image_file):

  splt = tf.strings.split(image_file, '/')

  # Read and decode an image file to a uint8 tensor
  input_image = tf.io.read_file(image_file)
  input_image = tf.io.decode_png(input_image)

  input_image = tf.image.grayscale_to_rgb(input_image)
  input_image = input_image * 40

  real_image = tf.io.read_file(PATH + splt[-3] +'/images/'+splt[-1]) 
  real_image = tf.io.decode_png(real_image)

  # Resize
  input_image = tf.image.resize(input_image, (470,470))
  real_image = tf.image.resize(real_image, (470,470))

  input_size = 470
  real_size = 470

  # Crop
  input_image = tf.image.crop_to_bounding_box(input_image,int((input_size-256)/2),int((input_size-256)/2), 256,256)
  real_image = tf.image.crop_to_bounding_box(real_image,int((real_size-256)/2),int((real_size-256)/2), 256,256)

  # Convert image to float32 tensor
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def load_single(image_file):
 
  input_image = tf.io.read_file(image_file)
  input_image = tf.io.decode_png(input_image)

  if input_image.shape[2] == 1:
    input_image = tf.image.grayscale_to_rgb(input_image)

  elif input_image.shape[2] == 4:
    input_image = tfio.experimental.color.rgba_to_rgb(input_image)

  input_image = input_image * 40


  # Crop
  input_image = tf.image.crop_to_bounding_box(input_image,50,0, 256,256)

  # Convert image to float32 tensor
  input_image = tf.cast(input_image, tf.float32)

  return input_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.show()

def test_dataset():
  test_dataset = tf.data.Dataset.list_files(PATH + 'test/labels/*.png')
  test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(BATCH_SIZE)

  generator = tf.keras.models.load_model('model/model.h5')

  for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

def test_single():
  image_file='images/seg_1rgb.png'

  input_image = load_single(image_file)
  input_image = resize_single(input_image,IMG_HEIGHT,IMG_WIDTH)
  input_image = normalize_single(input_image)
  input_image = input_image[None,:,:,:]


  plt.plot(1)
  plt.imshow(input_image[0] * 0.5 + 0.5)
  plt.title('a')

  plt.show()


  generator = tf.keras.models.load_model('model/model.h5')
  prediction = generator(input_image, training=True)

  plt.plot(1)
  plt.imshow(prediction[0]* 0.5 + 0.5)
  plt.title('a')

  plt.show()

def test_big_picture():
  image_file='images/seg_1rgb.png'

  input_image = tf.io.read_file(image_file)
  input_image = tf.io.decode_png(input_image)

  buffer_image = np.zeros_like(input_image, dtype=np.float32)

  images = []

  X = input_image.shape[0] / 256
  Y = input_image.shape[1] / 256


  for i in range(int(X)):
    for j in range(int(Y)):
      cropped_image = tf.image.crop_to_bounding_box(input_image,i*IMG_HEIGHT,j*IMG_WIDTH, IMG_HEIGHT,IMG_WIDTH)
      cropped_image = cropped_image * 40
      cropped_image = tf.cast(cropped_image, tf.float32)
      cropped_image = (cropped_image / 127.5) - 1
      cropped_image = cropped_image[None,:,:,:]
      images.append(cropped_image)

  # for img in images:
  #   plt.plot(1)
  #   plt.imshow(img[0] * 0.5 + 0.5)
  #   plt.title('a')

  #   plt.show()
      
  generator = tf.keras.models.load_model('model/model.h5')

  predictions = []
  for img in images:
    prediction = generator(img, training=True)
    predictions.append(prediction[0] * 0.5 + 0.5)
  
  predictions.reverse()

  # for img in predictions:
  #   plt.plot(1)
  #   plt.imshow(img)
  #   plt.title('a')

  #   plt.show()

  for i in range(int(X)):
    for j in range(int(Y)):
      img =  predictions.pop()
      img_array = img.numpy()
      buffer_image[i*IMG_HEIGHT:(i+1)*IMG_HEIGHT, j*IMG_WIDTH:(j+1)*IMG_WIDTH, :] = img_array

  plt.plot(1)
  plt.imshow(buffer_image)
  plt.title('a')

  plt.show()


def test_scaled_picture():
  image_file='images/seg_1rgb.png'

  input_image = tf.io.read_file(image_file)
  input_image = tf.io.decode_png(input_image)

  input_image = tf.cast(input_image, tf.float32)

  width = input_image.shape[0]
  height = input_image.shape[1]
  new_size = 1280

  input_image = tf.image.crop_to_bounding_box(input_image,int((width-new_size)/2),int((height-new_size)/2), new_size,new_size)
  input_image = resize_single(input_image, 256,256)  

  # plt.plot(1)
  # plt.imshow(input_image)
  # plt.title('a')

  # plt.show()

  input_image = (input_image / 127.5) - 1
  input_image = input_image[None,:,:,:]


  generator = tf.keras.models.load_model('model/model.h5')
  prediction = generator(input_image, training=True)


  plt.plot(1)
  plt.imshow(prediction[0]*0.5+0.5)
  plt.title('a')

  plt.show()


test_scaled_picture()