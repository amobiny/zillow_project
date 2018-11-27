


def write_spec(args):
    config_file = open(args.modeldir + args.run_name + '/config.txt', 'w')
    config_file.write('model: ' + args.run_name + '\n')
    config_file.write('input_dimension: ' + str(args.input_dim) + '\n')
    config_file.write('num_hidden_layers: ' + str(args.num_hidden_layers) + '\n')
    config_file.write('num_hidden_units: ' + str(args.hidden_units) + '\n')
    config_file.write('L2_regularization: ' + str(args.add_reg) + '\n')
    if args.add_reg:
        config_file.write('lambda: ' + str(args.lmbda) + '\n')
    config_file.write('optimizer: ' + 'Adam' + '\n')
    config_file.write('learning_rate: ' + str(args.init_lr) + ' : ' + str(args.lr_min) + '\n')
    config_file.write('batch_size: ' + str(args.batch_size) + '\n')
    config_file.write('keep_prob: ' + str(args.keep_prob) + '\n')
    config_file.close()




