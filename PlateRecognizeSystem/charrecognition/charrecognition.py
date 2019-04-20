import tensorflow as tf
import numpy as np
import os
from PIL import Image
import tensorflow.contrib.slim as slim
import charrecognition.environment as env
import time

def model(x, keep_prob):
    x_norm = x / 255.0


    net = slim.conv2d(x_norm, 32, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.conv2d(net, 64, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))

    net = slim.conv2d(net, 128, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.conv2d(net, 256, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.flatten(net)
    net = tf.nn.dropout(net, keep_prob)
    logits = slim.fully_connected(net, env.class_num)


    prob = tf.nn.softmax(logits, name=env.softmax_name)

    return logits, prob

class CharRecognition:
    def __init__(self, ckpt_path):
        graph = tf.Graph()
        self.ckpt_path= ckpt_path
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, env.image_size[1], env.image_size[0], env.image_size[2]], name=env.input_name)
            Y = tf.placeholder(tf.int64, [None], name=env.label_name)
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.logit, self.prob = model(self.X, env.dropout_rate)
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored')

    def __del__(self):
        print("Character Classifier is closed")
        self.sess.close()

    def predict(self, image):
        if image.format != 'RGB':
            image = image.convert('RGB')
        image = image.resize((env.image_size[0], env.image_size[1]))
        image = np.array(image)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            print('Expand_dims was executed.', image.shape)
        feed_dict = {self.X: image}
        argmax = tf.argmax(self.logit, -1)
        char_value = self.sess.run(argmax, feed_dict)

        return char_value[0]

    def predict_proba(self, image):
        if image.format != 'RGB':
            image = image.convert('RGB')
        image = image.resize((env.image_size[0], env.image_size[1]))
        image = np.array(image)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            print('Expand_dims was executed.', image.shape)
        feed_dict = {self.X: image}
        proba = self.sess.run(self.prob, feed_dict=feed_dict)
        return proba[0]

    def close(self):
        self.sess.close()
        print("Session Close")
