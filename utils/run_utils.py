



def write_spec(args):
    config_file = open(args.modeldir + args.run_name + '/config.txt', 'w')
    config_file.write('model: ' + args.run_name + '\n')
    config_file.write('optimizer: ' + 'Adam' + '\n')
    config_file.write('learning_rate: ' + str(args.init_lr) + ' : ' + str(args.lr_min) + '\n')
    config_file.write('batch_size: ' + str(args.batch_size) + '\n')
    config_file.write('keep_prob: ' + str(args.keep_prob) + '\n')
    config_file.close()




