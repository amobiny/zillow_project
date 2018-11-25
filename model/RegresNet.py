import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np

from model.ops import fc_layer, relu, dropout


class RegresNet(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        assert len(conf.hidden_units) == conf.num_hidden_layers, \
            'Number of hidden layers and hidden units does not match'
        self.input_shape = [None, self.conf.input_dim]
        self.output_shape = [None]
        self.create_placeholders()
        self.build_network()
        self.configure_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.labels_pl = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.keep_prob_pl = tf.placeholder(tf.float32)

    def build_network(self):
        self.summary_list = []
        x, s_list = fc_layer(self.inputs_pl, self.conf.hidden_units[0], 'FC1', self.conf.add_reg, self.conf.lmbda)
        self.summary_list.append(s_list)
        x = relu(x)
        x = dropout(x, 1 - self.keep_prob_pl, self.is_training_pl)
        for i in range(self.conf.num_hidden_layers - 1):
            x, s_list = fc_layer(x, self.conf.hidden_units[i], 'FC'+str(i+2), self.conf.add_reg, self.conf.lmbda)
            self.summary_list.append(s_list)
            x = relu(x)
            x = dropout(x, 1 - self.keep_prob_pl, self.is_training_pl)
        self.y_pred, s_list = fc_layer(x, 1, 'OUT', self.conf.add_reg, self.conf.lmbda)
        self.summary_list.append(s_list)

    def loss_func(self):
        with tf.name_scope('Loss'):
            if self.conf.loss_type == 'mse':
                with tf.name_scope('mse'):
                    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels_pl,
                                                                       predictions=tf.squeeze(self.y_pred)))
            if self.conf.add_reg:
                with tf.name_scope('L2_loss'):
                    l2_loss = tf.reduce_sum(
                        self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
                    self.total_loss = loss + l2_loss
            else:
                self.total_loss = loss
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def configure_network(self):
        self.loss_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=2000,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss)] + self.summary_list
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_loss = 1000
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        from DataLoader import DataLoader
        self.data_reader = DataLoader(self.conf)
        self.num_train_batch = int(self.data_reader.num_tr / self.conf.batch_size)
        self.num_val_batch = int(self.data_reader.num_val / self.conf.val_batch_size)
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                glob_step = epoch * self.num_train_batch + train_step
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                feed_dict = {self.inputs_pl: x_batch, self.labels_pl: y_batch,
                             self.is_training_pl: True, self.keep_prob_pl: self.conf.keep_prob}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss = self.sess.run(self.mean_loss)
                    self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}'.format(train_step, loss))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op], feed_dict=feed_dict)
            self.evaluate(glob_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.inputs_pl: x_val, self.labels_pl: y_val,
                         self.is_training_pl: False, self.keep_prob_pl: 1}
            self.sess.run(self.mean_loss_op, feed_dict=feed_dict)
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss = self.sess.run(self.mean_loss)
        self.save_summary(summary_valid, train_step + self.conf.reload_step, mode='valid')
        if valid_loss < self.best_validation_loss:
            self.best_validation_loss = valid_loss
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''

        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, {2}'
              .format(train_step, valid_loss, improved_str))

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')



