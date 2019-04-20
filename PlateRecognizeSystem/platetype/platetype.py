import tensorflow as tf
import numpy as np
import os
from PIL import Image
import tensorflow.contrib.slim as slim
import platetype.environment as env
import time

def model(x, keep_drop=1.0):
    #x = tf.cast(x, tf.float32)
    x_norm = tf.divide(x, 255.0)
    net = slim.conv2d(x_norm, 32, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 64 32

    net = slim.conv2d(net, 64, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 16

    net = slim.conv2d(net, 128, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 16 8

    net = slim.conv2d(net, 256, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 8 4
    net = tf.nn.dropout(net, keep_prob=keep_drop)

    net = slim.conv2d(net, 512, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 4 2

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))

    net = slim.flatten(net)
    net = tf.nn.dropout(net, keep_prob=keep_drop)

    net_t = slim.fully_connected(net, 8)
    net_t_soft = tf.nn.softmax(net_t, name=env.softmax_name)
    net_c = slim.fully_connected(net, 2)
    net_c_soft = tf.nn.softmax(net_c)
    return net_t, net_c, net_t_soft, net_c_soft

class PlateType:
    def __init__(self, ckpt_path):
        graph = tf.Graph()
        self.ckpt_path = ckpt_path
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, env.image_size[1], env.image_size[0], env.image_size[2]], name=env.input_name)
            Y = tf.placeholder(tf.int64, [None], name=env.label_name)
            Y_p = tf.where(Y > 0, tf.ones_like(Y), tf.zeros_like(Y))
            self.y_hat_t, _, self.type_proba, _ = model(self.X)
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
        argmax = tf.argmax(self.y_hat_t, -1)

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
        proba = self.sess.run(self.type_proba, feed_dict=feed_dict)
        return proba[0]

    def close(self):
        self.sess.close()
        print("Session Close")
