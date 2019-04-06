import argparse
import tensorflow as tf
import time


class Prodector:
    def __init__(self, input_filename, model_path):
        self.input_filename = input_filename
        self.model_path = model_path

        # FIXME 应该在加载图片后从图片中读。
        self.img_w = 208
        self.img_h = 208

        self.n_classes = 4

        self.init_model()

    def init_model(self):
        self.images = tf.placeholder(tf.float32, [1, self.img_w, self.img_h, 3])

        # 模型结构定义
        # layer1
        with tf.variable_scope('conv1') as scope:
            weights_1 = tf.get_variable('weights', shape=[5, 5, 3, 32], dtype=tf.float32)
            biases_1 = tf.get_variable('biases', shape=[32], dtype=tf.float32)
            conv = tf.nn.conv2d(self.images, weights_1, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases_1)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
        with tf.variable_scope('pooling1') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

        # layer2
        with tf.variable_scope('conv2') as scope:
            weights_2 = tf.get_variable('weights', shape=[5, 5, 32, 64], dtype=tf.float32)
            biases_2 = tf.get_variable('biases', shape=[64], dtype=tf.float32)
            conv = tf.nn.conv2d(pool1, weights_2, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases_2)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
        with tf.variable_scope('pooling2') as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

        # layer3
        with tf.variable_scope('local3') as scope:
            # -1代表不用指定这一维的大小，函数会自动计算
            reshape = tf.reshape(pool2, shape=[1, -1])
            # 获得reshape的列数，矩阵点乘要满足列数等于行数
            dim = reshape.get_shape()[1].value
            weights_3 = tf.get_variable('weights', shape=[dim, 128], dtype=tf.float32)
            biases_3 = tf.get_variable('biases', shape=[128], dtype=tf.float32)
            local3 = tf.nn.relu(tf.matmul(reshape, weights_3) + biases_3, name=scope.name)

        # layer4
        with tf.variable_scope('local4') as scope:
            weights_4 = tf.get_variable('weights', shape=[128, 84], dtype=tf.float32)
            biases_4 = tf.get_variable('biases', shape=[84], dtype=tf.float32)
            local4 = tf.nn.relu(tf.matmul(local3, weights_4) + biases_4, name=scope.name)

        # layer5
        with tf.variable_scope('output_layer') as scope:
            weights_5 = tf.get_variable('weights', shape=[84, self.n_classes], dtype=tf.float32)
            biases_5 = tf.get_variable('biases', shape=[self.n_classes], dtype=tf.float32)
            self.output = tf.nn.softmax(tf.matmul(local4, weights_5) + biases_5, name=scope.name)
            self.output = tf.argmax(self.output, 1)
        self.sess = tf.Session()
        # 载入模型参数
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))

    def prodect(self):
        start = time.perf_counter()  # 开始计时
        image_contents = tf.read_file(self.input_filename)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.image.resize_images(image, (self.img_w, self.img_h), method=1)
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, 0)  # 将image的维度转换为(1，img_w，img_h，3)
        prediction = self.sess.run(self.output, feed_dict={self.images: self.sess.run(image)})
        print("prediction is : {}".format(prediction))


def init_args():
    parser = argparse.ArgumentParser(description='model args.')
    parser.add_argument('-i', dest='input_filename', metavar='filename', help="input image file name.")
    parser.add_argument('-m', dest='model_path', metavar='path', help="model file path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()
    prodector = Prodector(args.input_filename, args.model_path)
    prodector.prodect()
