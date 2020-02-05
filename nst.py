import os
import time

from PIL import Image
from nst_utils import *

import tensorflow as tf

import imageio as img
import redis as rd


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = (tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))) / (4 * n_H * n_W * n_C)
    
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S), shape=[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G), shape=[n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (tf.reduce_sum(tf.square(tf.subtract(GS, GG)))) / (4 * n_C**2 * n_H**2 * n_W**2)
    
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha * J_content + beta * J_style
    
    return J


if __name__ == '__main__':
    #
    # Define all constants and parameters
    #
    # Prepare all possible weights of hidden layers
    BASE_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    TARGET_WEIGHTS = []

    for w1 in BASE_WEIGHTS:
        for w2 in BASE_WEIGHTS:
            for w3 in BASE_WEIGHTS:
                for w4 in BASE_WEIGHTS:
                    for w5 in BASE_WEIGHTS:
                        if (w1 + w2 + w3 + w4 + w5 == 1.):
                            TARGET_WEIGHTS.append((w1, w2, w3, w4, w5))

    print("DEBUG: number of targets:", len(TARGET_WEIGHTS))

    # Input folder, content image file name and style image file name
    INPUT_FOLDER = os.environ["INPUT_FOLDER"]
    CONTENT_IMAGE = os.environ["CONTENT_IMAGE"]
    STYLE_IMAGE = os.environ["STYLE_IMAGE"]
    
    # Hidden layer to use
    HIDDEN_LAYER = os.environ["HIDDEN_LAYER"]

    # Number of iterations
    ITERATIONS = 200
    
    # Redis Service
    REDIS_HOST = os.environ["REDIS_SERVICE_HOST"]
    REDIS_PORT = os.environ["REDIS_SERVICE_PORT"]
    REDIS_PASS = os.environ["REDIS_PASSWORD"]
    
    # Where to save results
    OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"]
    
    #
    # Find the next available set of weights checking with Redis
    #
    # Open connection to Redis
    r = rd.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS)
    
    for (w1, w2, w3, w4, w5) in TARGET_WEIGHTS:
        STYLE_LAYERS = [
            ('conv1_1', w1),
            ('conv2_1', w2),
            ('conv3_1', w3),
            ('conv4_1', w4),
            ('conv5_1', w5)]
        
        image_name = "result_" + HIDDEN_LAYER + "_" + str(w1) + "_" + str(w2) + "_" + str(w3) + "_" + str(w4) + "_" + str(w5) + ".jpg"
        
        if (r.exists(image_name) == 0):
            r.set(image_name, 1)
            r.expire(image_name, 1200)
            print(f"Key {image_name} set")
            start_time = time.time()
            break
    
    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()
     
    # Load content image
    content_image = img.imread(INPUT_FOLDER + "/" + CONTENT_IMAGE)
    content_image = reshape_and_normalize_image(content_image)
    
    # Load style image
    style_image = img.imread(INPUT_FOLDER + "/" + STYLE_IMAGE)
    style_image = reshape_and_normalize_image(style_image)

    # Add noise to content image
    generated_image = generate_noise_image(content_image)
    
    # Load the VGG19 model
    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    
    # Select the output tensor of some layer
    out = model[HIDDEN_LAYER]

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)
    
    # Compute total cost
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    
    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)
    
    #
    # Make the picture
    #

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(generated_image))
    
    for i in range(ITERATIONS):
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

    # save last generated image
    save_image(OUTPUT_FOLDER + "/" + image_name, generated_image)
    r.set(image_name, 1)
    total_time = time.time() - start_time
    print(f"Image {image_name} processed in {total_time} seconds.")
        
    # Clear everything
    tf.reset_default_graph()
    sess.close()
    

