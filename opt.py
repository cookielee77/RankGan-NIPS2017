import logging
import argparse
import os
import time
import numpy as np


class Options(object):
    def __init__(self):
        self.seq_len = 20
        self.vocab_size = 5000
        # generator 
        self.g_emb_dim = 32 # embedding dimension for generator
        self.g_hid_dim = 32 # hidden dimension for generator
        self.start_token = 0
        self.generated_num = 10000 # sample positive files number
        # ranker
        self.rank_emb_dim = 64 # embedding dimension for ranker
        self.rank_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.rank_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.dropout_ratio = 0.75
        self.gamma = 1 # temprature control parameters
        self.num_class = 2
        # file path
        self.target_path = 'save/target_params.pkl'
        self.positive_file = 'save/real_data.txt'
        self.negative_file = 'save/generator_sample.txt'
        self.eval_file = 'save/eval_file.txt'


def create_logging(FLAGS):
    # head = '%(asctime)-15s %(message)s'
    head = ''
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log_file = '{}_{}.log'.format(FLAGS.prefix, time.strftime('%Y-%m-%d-%H-%M'))
    logging.basicConfig(filename=os.path.join('./log', log_file), level=logging.DEBUG, format=head)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logging.info('start with arguments %s', FLAGS)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
