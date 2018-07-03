"""
Implementation of Neural Style Transfer in Keras
Source: https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216
Original Source Code: https://github.com/walid0925/AI_Artistry
"""

# IMPORTS/ IMAGE PROCESSING ################################################################################
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
import time

start_time = time.time()

target_height = 512
target_width = 512
target_size = (target_height, target_width)
content_image_path = "sources/cathedral1.jpg"
style_image_path = "styles/giger_crop.jpg"
gen_image_output_path = './results/output.jpg'

# Turns content image into an array (with the magic of Keras)
content_image = load_img(path=content_image_path, target_size=target_size)
content_image_orig_size = content_image.size
content_image_array = img_to_array(content_image)
content_image_array = K.variable(preprocess_input(np.expand_dims(content_image_array, axis=0)), dtype='float32')

# Turns content image into an array (with the magic of Keras)
style_image = load_img(path=style_image_path, target_size=target_size)
style_image_array = img_to_array(style_image)
style_image_array = K.variable(preprocess_input(np.expand_dims(style_image_array, axis=0)), dtype='float32')

# Initializes the generated image with random pixel values
gen_image = np.random.randint(256, size=(target_width, target_height, 3)).astype('float64')
gen_image = preprocess_input(np.expand_dims(gen_image, axis=0))
gen_image_placeholder = K.placeholder(shape=(1, target_width, target_height, 3))


# Function that gets the feature representation of input x for one or more layers
def getFeatureReps(x, layer_names, model):
    feature_matrices = []
    for ln in layer_names:
        selected_layer = model.get_layer(ln)
        raw_features = selected_layer.output
        raw_features_shape = K.shape(raw_features).eval(session=tf_session)
        N_l = raw_features_shape[-1]
        M_l = raw_features_shape[1] * raw_features_shape[2]
        feature_matrix = K.reshape(raw_features, (M_l, N_l))
        feature_matrix = K.transpose(feature_matrix)
        feature_matrices.append(feature_matrix)
    
    return feature_matrices


# LOSS FUNCTIONS ##########################################################################################

# Function that gets the content loss
def getContentLoss(F, P):
    content_loss = 0.5 * K.sum(K.square(F - P))
    return content_loss


# Function that gets the Gram matrix of a given matrice
def makeGramMatrix(matrix):
    gram_matrix = K.dot(matrix, K.transpose(matrix))
    return gram_matrix


# Function that calculates the style loss
def getStyleLoss(weights, gram_matrices, matrices):
    style_loss = K.variable(0.)
    for weight, gram_matrix, matrix in zip(weights, gram_matrices, matrices):
        M_l = K.int_shape(gram_matrix)[1]
        N_l = K.int_shape(gram_matrix)[0]
        G_gram = makeGramMatrix(gram_matrix)
        A_gram = makeGramMatrix(matrix)
        style_loss += weight * 0.25 * K.sum(K.square(G_gram - A_gram)) / (N_l**2 * M_l**2)

    return style_loss


# Function that calculates the total loss
def getTotalLoss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = getFeatureReps(gImPlaceholder, layer_names=[content_layer_name], model=gModel)[0]
    Gs = getFeatureReps(gImPlaceholder, layer_names=style_layer_names, model=gModel)
    contentLoss = getContentLoss(F, P)
    styleLoss = getStyleLoss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss


# Function that calculates the total loss (style and content)
def calculateLoss(generated_img_array):
    if generated_img_array.shape != (1, target_width, target_width, 3):
        generated_img_array = generated_img_array.reshape((1, target_width, target_height, 3))
    loss_fcn = K.function([gModel.input], [getTotalLoss(gModel.input)])
    return loss_fcn([generated_img_array])[0].astype('float64')


# POST-TRANSFER (RE)PROCESSSING ###########################################################################

# Function that processes the array after style transfer
def postprocessArray(x):
    # Zero-center by mean pixel
    if x.shape != (target_width, target_height, 3):
        x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


# Function that reprocess the array 
def reprocessArray(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x

def saveOriginalSize(x, target_size=content_image_orig_size):
    xIm = Image.fromarray(x)
    xIm = xIm.resize(target_size)
    xIm.save(gen_image_output_path)
    return xIm

# Function that calculates the gradient of the image with respect to generated image
def getGradient(generated_img_array):
    if generated_img_array.shape != (1, target_width, target_height, 3):
        generated_img_array = generated_img_array.reshape((1, target_width, target_height, 3))

    grad_fcn = K.function([gModel.input], K.gradients(getTotalLoss(gModel.input), [gModel.input]))
    grad = grad_fcn([generated_img_array])[0].flatten().astype('float64')
    return grad


# NEURAL NETWORK STYLE TRANSFER BELOW #####################################################################

# Neural network constructed below (built on top of pretrained VGG16 conv. network)
tf_session = K.get_session()
cModel = VGG16(include_top=False, weights='imagenet', input_tensor=content_image_array)
sModel = VGG16(include_top=False, weights='imagenet', input_tensor=style_image_array)
gModel = VGG16(include_top=False, weights='imagenet', input_tensor=gen_image_placeholder)
content_layer_name = 'block4_conv2'
style_layer_names = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1',]

P = getFeatureReps(x=content_image_array, layer_names=[content_layer_name], model=cModel)[0]
As = getFeatureReps(x=style_image_array, layer_names=style_layer_names, model=sModel)
ws = np.ones(len(style_layer_names)) / float(len(style_layer_names))

iterations = 600
x_val = gen_image.flatten()
xopt, f_val, info= fmin_l_bfgs_b(calculateLoss, x_val, fprime=getGradient, maxiter=iterations, disp=True, iprint=50)


# IMAGE OUTPUT #############################################################################################

# Turns the array back into a 
xOut = postprocessArray(xopt)
xIm = saveOriginalSize(xOut)
print('Image saved')
end = time.time()
print('Time taken: {}'.format(end-start_time))