import tensorflow as tf
import numpy as np
import os
from PIL import Image
import tensorflow.contrib.slim as slim
import regionrecognition.environment as env
import time

def model(x, keep_prob=1.0):
    x_norm = x / 255.0

    with tf.name_scope('layer1'):
        net = slim.conv2d(x_norm, 64, kernel_size=(3,3))
        net = slim.max_pool2d(net, (2,2))

    with tf.name_scope('layer2'):
        net = slim.conv2d(net, 128, kernel_size=(3,3))
        net = slim.max_pool2d(net, (2,2))
    with tf.name_scope('layer3'):
        net = slim.conv2d(net, 512, kernel_size=(3,3))
        net = slim.max_pool2d(net, (2,2))

    with tf.name_scope('layer4'):
        net = slim.conv2d(net, 1024, kernel_size=(3,3))
        net = slim.max_pool2d(net, (2,2))


    with tf.name_scope('output'):
        net = slim.flatten(net)
        net = tf.nn.dropout(net, keep_prob)
        logits = slim.fully_connected(net, env.class_num)


    prob = tf.nn.softmax(logits, name='hypothesis')

    return logits, prob

class RegionRecognition:
    def __init__(self, ckpt_path):
        graph = tf.Graph()
        self.ckpt_path = ckpt_path
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, env.image_size[1], env.image_size[0], env.image_size[2]], name=env.input_name)
            Y = tf.placeholder(tf.int64, [None], name=env.label_name)
            P = tf.placeholder(tf.float32)
            self.y_hat, self.y_proba = model(self.X)
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored')

    def __del__(self):
        print("Type Classifier is closed")
        self.sess.close()

    def predict(self, image):
        image = image.resize((env.image_size[0], env.image_size[1]))
        image = np.array(image)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            print('Expand_dims was executed.', image.shape)
        feed_dict = {self.X: image}
        argmax = tf.argmax(self.y_hat, -1)

        start = time.time()
        type_value = self.sess.run(argmax, feed_dict)

        return type_value[0]

    def predict_proba(self, image):
        image = image.resize((env.image_size[0], env.image_size[1]))
        image = np.array(image)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            print('Expand_dims was executed.', image.shape)
        feed_dict = {self.X: image}
        proba = self.sess.run(self.y_proba, feed_dict=feed_dict)
        return proba[0]

    def close(self):
        self.sess.close()
        print("Session Close")
