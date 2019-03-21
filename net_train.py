# 训练网络主程序

import os
import numpy as np
import tensorflow as tf
import read_data
import net_model

N_CLASSES = 4       # 分类个数
IMG_W = 208
IMG_H = 208         # 图像的尺寸
BATCH_SIZE = 20
CAPACITY = 8000     # 队列最大容量
MAX_STEP = 40001     # 训练的迭代次数
learning_rate = 1e-5  # 学习率


# 定义开始训练的函数
def run_training():
    # 输出文件的位置
    logs_train_dir = 'E:/traffic_sign_detection/output'
    # 调用input_data文件的get_files()函数获得image_list, label_list
    train, train_label = read_data.get_files()
    # 获得image_batch, label_batch
    train_batch, train_label_batch = read_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    # 进行前向训练，获得回归值
    train_prediction = net_model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    # 计算获得损失值loss
    train_loss = net_model.losses(train_prediction, train_label_batch)
    # 对损失值进行优化
    train_op = net_model.training(train_loss, learning_rate)
    # 根据计算得到的损失值，计算出分类准确率
    train_acc = net_model.evaluation(train_prediction, train_label_batch)
    # 将图形、训练过程合并在一起
    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    # 将训练日志写入到文件夹内
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 保存变量
    saver = tf.train.Saver()
    # 执行训练过程，初始化变量
    sess.run(tf.global_variables_initializer())
    # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    # 启动入队的线程，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            # 使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
            # 会抛出一个 OutOfRangeError 的异常，这时候就应该停止Session中的所有线程了;
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            # 打印一次损失值和准确率
            if step % 50 == 0:
                print('Step{}, loss={:.5f}, accuracy={:.4f}'.format(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            # 保存一次训练得到的模型
            if (step + 1) == MAX_STEP or step == 20000 or step == 30000:
                checkpoint_path = os.path.join(logs_train_dir, 'net_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    # 如果读取到文件队列末尾会抛出此异常
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()       # 使用coord.request_stop()来发出终止所有线程的命令
    coord.join(threads)            # coord.join(threads)把线程加入主线程，等待threads结束
    sess.close()                   # 关闭会话


def main():
    run_training()


if __name__ == '__main__':
    main()
