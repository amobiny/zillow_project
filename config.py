import tensorflow as tf

flags = tf.app.flags


flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_integer('reload_Epoch', 0, 'Reload the model trained for Epoch epochs')

flags.DEFINE_integer('max_epoch', 3000, '# of step for training')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_string('loss_type', 'mse', 'Loss type; either mse or mae')
flags.DEFINE_boolean('add_reg', True, 'Use L2 regularization on network parameters')
flags.DEFINE_float('lmbda', 5e-4, 'L2 regularization coefficient')
flags.DEFINE_integer('batch_size', 20, 'training batch size')
flags.DEFINE_integer('val_batch_size', 100, 'training batch size')
flags.DEFINE_integer('test_batch_size', 27, 'test batch size')
flags.DEFINE_float('keep_prob', 0.7, 'Probability of keeping a unit in drop-out')
flags.DEFINE_boolean('normalize', True, 'Whether to normalize the data or not')
flags.DEFINE_integer('num_hidden_layers', 2, 'Number of hidden fully-connected layers')
flags.DEFINE_list('hidden_units', [100, 10], 'number of hidden units')

# data
flags.DEFINE_string('data_dir', '/data/', 'Data directory')
flags.DEFINE_integer('input_dim', 18, 'Dimension of the input data')

# Saving results
flags.DEFINE_string('run_name', 'run4', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')


args = tf.app.flags.FLAGS


