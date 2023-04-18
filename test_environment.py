import tensorflow as tf
from numpy import asarray
from numpy import load
import time
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

print('hello')

FACE_MODEL='models/facenet_keras_128.h5'
IMAGE_PATH = 'images/salah.jpg'
MODEL_NAME='facenet'

tick = time.time()

# load the face recognition model
facenet_128_model = tf.keras.models.load_model(FACE_MODEL)
print('Time to load facenet model and dataset: ', time.time()-tick, "sec")
print('**************** Loaded Model ****************')



def read_image(image_path):
  image = Image.open(image_path)
  # convert to RGB, if needed
  image = image.convert('RGB')
  # convert to numpy array
  pixels = np.asarray(image)
  return pixels

image=read_image(IMAGE_PATH)

detector = MTCNN()
faces_metadata = detector.detect_faces(image) 
print ('number of detected faces: ', len(faces_metadata))

def get_embedding_facenet(face_pixels, model):
    #print('get_embedding_facenet')
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    embedding = model.predict(samples)
    return embedding[0]




def get_face(faces_metadata, face_index=0):
    x0 = int(faces_metadata[face_index]['box'][0])
    y0 = int(faces_metadata[face_index]['box'][1])
    w0 = int(faces_metadata[face_index]['box'][2])
    h0 = int(faces_metadata[face_index]['box'][3])
    print('x=' , x0)
    required_size=(160,160)
    #crop image 
    face = image[y0:y0+h0, x0:x0+w0]
    # resize pixels to the model size
    face = Image.fromarray(face)
    #resize the image 
    face = face.resize(required_size)
    #convert the image to numpy array structure
    face = np.asarray(face)
    return face


face = get_face(faces_metadata, face_index=0)
print(face.shape)

face_embedding = get_embedding_facenet(face, facenet_128_model)

print(face_embedding.shape)