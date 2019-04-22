# 读取图像和GroundTruth数据

import tensorflow as tf
import numpy as np

import os


def get_files(label_file_name, dataset_path):
    # 读取GroundTruth中的数据作为标签
    f = open(label_file_name, 'r')
    line = f.readline()
    data_list = []
    while line:
        line = os.path.join(dataset_path, line)
        line = line.replace('\n', '')
        line = line.replace('warning', '0')
        line = line.replace('prohibitory', '1')
        line = line.replace('mandatory', '2')
        line = line.replace('indication', '3')
        num = list(line.split(';'))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)
    image_list = list(data_array[:, 0])
    label_list = []
    for label in data_array[:, 5]:
        real_label = [0] * 4
        real_label[int(label)] = 1
        label_list.append(real_label)
    return image_list, label_list


def get_batch(image_list, label_list, image_W, image_H, batch_size, capacity):
    """
    Args:
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    # 数据转换
    image = tf.cast(image_list, tf.string)  # 将image数据转换为string类型
    label = tf.cast(label_list, tf.int32)  # 将label数据转换为int类型

    # 生成输入的队列，每次在数据集中产生一个切片
    input_queue = tf.train.slice_input_producer([image, label])
    # 标签为索引为1的位置
    label = input_queue[1]
    # 图片的内容为读取索引为0的位置所得的内容
    image_contents = tf.read_file(input_queue[0])
    # 解码图像，解码为一个张量
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 对图像的大小进行调整，调整大小为image_W,image_H
    image = tf.image.resize_images(image, (image_W, image_H), method=1)
    # 对图像进行标准化
    image = tf.image.per_image_standardization(image)
    # 使用train.batch函数来组合样例，image和label代表训练样例和所对应的标签，batch_size参数
    # 给出了每个batch中样例的个数，capacity给出了队列的最大容量，当队列长度等于容量时，暂停入队
    # 只是等待出队
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacity,
                                                      min_after_dequeue=1)

    return image_batch, label_batch
