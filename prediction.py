# 调用训练好的模型预测

import tensorflow as tf
import RPi.GPIO as GPIO
import threading
import time
import sys
import os

save_dir = '/home/pi/traffic_sign_detection/models'   # 训练过的模型参数保存目录
store_dir = '/home/pi/traffic_sign_detection/'    # 拍摄图片存储目录
n_classes = 4
img_w = 208
img_h = 208
img_name = 'image.jpg'

img_num = 0    # 拍摄图片的总数

images = tf.placeholder(tf.float32, [1, img_w, img_h, 3])

class tt(threading.Thread):
    flag = 0
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        while True:
            input_k = str(sys.stdin.readline()).strip('\n')
            if input_k == 'q':
                self.flag = 1    # 子线程捕获q按下，主线程退出循环

def main():
    my_t = tt()
    my_t.start()
    global img_num
    
    while my_t.flag == 0:
        os.system('fswebcam --no-banner image.jpg')    # 树霉派通过板载摄像头获取图片
        img_num = img_num + 1
        image_contents = tf.read_file(store_dir + img_name)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.image.resize_images(image, (img_w, img_h), method=1)
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, 0)    # 将image的维度转换为(1，img_w，img_h，3)
        prediction = sess.run(output, feed_dict={images: sess.run(image)})

        # Warning状态下11引脚输出低电平，12引脚输出高电平
        # Prohibitory状态下11引脚输出高电平，12引脚输出高电平
        # Mandatory和Indication状态下11引脚输出低电平，12引脚输出低电平
        if prediction == 0:
            print('Warning!')
            GPIO.output(11, GPIO.LOW)
            GPIO.output(12, GPIO.HIGH)
        elif prediction == 1:
            print('Prohibitory!')
            GPIO.output(11, GPIO.HIGH)
            GPIO.output(12, GPIO.HIGH)
        elif prediction == 2:
            print('Mandatory!')
            GPIO.output(11, GPIO.LOW)
            GPIO.output(12, GPIO.LOW)
        else:
            print('Indication!')
            GPIO.output(11, GPIO.LOW)
            GPIO.output(12, GPIO.LOW)

# 模型结构定义
# layer1
with tf.variable_scope('conv1') as scope:
    weights_1 = tf.get_variable('weights', shape=[5, 5, 3, 32], dtype=tf.float32)
    biases_1 = tf.get_variable('biases', shape=[32], dtype=tf.float32)
    conv = tf.nn.conv2d(images, weights_1, strides=[1, 1, 1, 1], padding='SAME')
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
    weights_5 = tf.get_variable('weights', shape=[84, n_classes], dtype=tf.float32)
    biases_5 = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32)
    output = tf.nn.softmax(tf.matmul(local4, weights_5) + biases_5, name=scope.name)
    output = tf.argmax(output, 1)


sess = tf.Session()
# 载入模型参数
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(save_dir))

print('Start Predicting Results...')

# 11引脚和12引脚置为输出，初始化为低电平
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)

start = time.perf_counter()    # 开始计时
main()    # 启动主线程，开始预测，按下q键并回车停止
dur = time.perf_counter() - start    # 停止计时
per_dur = dur / img_num
GPIO.cleanup()    # 清理引脚

print('Average processing time per image is {:.2f}s'.format(per_dur))

sess.close()
