# 建立训练时的网络模型

import tensorflow as tf
regularizer = 0.0001
from tensorflow.python import keras


def inference(images, batch_size, n_classes):
    # tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现变量共享。下次调用不用重新产生，这样可以保存参数
    # layer1
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', shape=[5, 5, 3, 32], dtype=tf.float32,
                                  initializer=keras.initializers.he_normal(seed=None))
        tf.add_to_collection('loss_w', tf.contrib.layers.l2_regularizer(regularizer)(weights))
        biases = tf.get_variable('biases', shape=[32], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.tanh(pre_activation, name=scope.name)
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

    # layer2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[5, 5, 32, 64], dtype=tf.float32,
                                  initializer=keras.initializers.he_normal(seed=None))
        tf.add_to_collection('loss_w', tf.contrib.layers.l2_regularizer(regularizer)(weights))
        biases = tf.get_variable('biases', shape=[64], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.tanh(pre_activation, name=scope.name)
    with tf.variable_scope('pooling2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

    # layer3
    with tf.variable_scope('local3') as scope:
        # -1代表不用指定这一维的大小，函数会自动计算
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        # 获得reshape的列数，矩阵点乘要满足列数等于行数
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 128], dtype=tf.float32,
                                  initializer=keras.initializers.he_normal(seed=None))
        tf.add_to_collection('loss_w', tf.contrib.layers.l2_regularizer(regularizer)(weights))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)

    # layer4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[128, 84], dtype=tf.float32,
                                  initializer=keras.initializers.he_normal(seed=None))
        tf.add_to_collection('loss_w', tf.contrib.layers.l2_regularizer(regularizer)(weights))
        biases = tf.get_variable('biases', shape=[84], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.tanh(tf.matmul(local3, weights) + biases, name=scope.name)

    # layer5
    with tf.variable_scope('output_layer') as scope:
        weights = tf.get_variable('weights', shape=[84, n_classes], dtype=tf.float32,
                                  initializer=keras.initializers.he_normal(seed=None))
        tf.add_to_collection('loss_w', tf.contrib.layers.l2_regularizer(regularizer)(weights))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        output = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        #output1 = tf.argmax(output, 1, name='output1')
    return output


# 定义损失函数，定义传入值和标准值的差距
def losses(prediction, labels):
    with tf.variable_scope('loss') as scope:
        # prediction表示神经网络的输出结果，labels表示标准答案。先计算softmax回归值后计算交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels,
                                                                       name='x_entropy_per_example')
        # 求cross_entropy所有元素的平均值
        loss = tf.reduce_mean(cross_entropy, name='loss')
        loss = loss + tf.add_n(tf.get_collection('loss_w'))
        # 对loss值进行标记汇总，一般在画loss, accuracy时会用到这个函数。
        tf.summary.scalar(scope.name, loss)
    return loss


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 设置一个用于记录全局训练步骤的单值
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 最小化loss，并更新var_list，返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# 定义评价函数，返回准确率
def evaluation(prediction, labels):
    with tf.variable_scope('accuracy') as scope:
        # 计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
        # K表示每个样本的预测结果的前K个最大的数里面是否含有label中的值,k一般都是取1
        correct = tf.nn.in_top_k(prediction, labels, 1)
        # 转换类型
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        # 对准确度进行标记汇总
        tf.summary.scalar(scope.name, accuracy)
    return accuracy
