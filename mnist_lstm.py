import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


class HyperParameter(object):
    def __init__(self):
        self.batch_size = 64
        self.epoch = 5000
        self.lr = 0.001
        self.hidden_size = 1024
        self.optimizer = "Adam"
        self.dropout = 0.5
        self.padding = 'SAME'
        self.kernel_size = [5, 5]
        self.padding_size = [2, 2]
        self.shuffle = True
        self.n_classes = 10
        self.global_steps = tf.train.create_global_step(graph=None)
        self.base_path = os.path.join("./saver", str(int(time.time())))


class Lenet(HyperParameter):
    def __init__(self):
        super().__init__()
        self.x = tf.placeholder(tf.float32, [None, 784], name="X")
        self.x_image = tf.reshape(self.x, [-1, 28, 28], name="X_image")
        self.y = tf.placeholder(tf.float32, [None, 10], name="Y")
        self.keep_prob = tf.placeholder(tf.float32)
        self.layer = self.lstm_layers(self.x_image)
        self.out = self.full_layers(self.layer)
        self.loss = self.loss_op(self.out, self.y)
        self.accuracy = self.accuracy_op(self.out, self.y)
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.lr,
            optimizer=self.optimizer,
            name=self.optimizer
        )

    def lstm_layers(self, inputs):
        with tf.variable_scope("LSTM"):
            print(inputs.shape)
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_classes)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=inputs,
                initial_state=None,
                dtype=tf.float32,
                time_major=False,
            )
            return outputs

    def full_layers(self, inputs):
        with tf.variable_scope("Full"):
            # 对数据的三种不同的处理方式
            inputs = tf.layers.flatten(inputs)  # 1、拍扁：降成一维
            # inputs = inputs[:, -1, :]  # 2、取每个其中一片数据
            # inputs = tf.reduce_mean(inputs, 1)  # 3、取均值
            return tf.layers.dense(inputs, self.n_classes)  # 全连接映射

    @staticmethod
    def loss_op(logits, labels):
        with tf.variable_scope("Loss"):
            cross_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            return tf.reduce_mean(cross_loss)

    @staticmethod
    def accuracy_op(logits, labels):
        with tf.variable_scope("Accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('mnist/', one_hot=True)  # 加载数据
    model = Lenet()  # 加载模型
    saver_hook = tf.train.CheckpointSaverHook(  # 定义模型保存器
        checkpoint_dir=os.path.join(model.base_path, "checkpoints"),
        save_steps=100,
    )
    summary_hook = tf.train.SummarySaverHook(  # 定义日志保存器
        save_steps=100,
        output_dir=os.path.join(model.base_path, "summary"),
        summary_op=tf.summary.merge_all()
    )
    with tf.train.SingularMonitoredSession(hooks=[summary_hook, saver_hook]) as sess:  # 不需要变量初始化
        for i in range(1, model.epoch + 1):
            x_train, y_train = mnist.train.next_batch(model.batch_size)
            _, l, acc = sess.run([model.train_op, model.loss, model.accuracy],
                                 feed_dict={model.x: x_train, model.y: y_train, model.keep_prob: model.dropout,
                                            })
            if i % 100 == 0:
                print("Epoch:%d\t--\tLoss:%.3f\t--\tAcc:%.2f" % (i, l, acc))
            if i % 1000 == 0:
                train_ac = sess.run([model.accuracy],
                                    feed_dict={model.x: mnist.test.images,
                                               model.y: mnist.test.labels,
                                               model.keep_prob: 1.0})
                print('测试集Acc:{0}'.format(train_ac[0]))
