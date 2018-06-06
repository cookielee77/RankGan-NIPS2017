import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Rank_Data_loader
from generator import Generator
from ranker import Ranker
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
from opt import *
import cPickle
import logging
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


SEED = 88

parser = argparse.ArgumentParser()
parser.add_argument('--gen_pre_batch_size', type = int, default = 64, help = 'batch size for generator in pre-training')
parser.add_argument('--gen_batch_size', type = int, default = 32, help = 'batch size for generator in ad learning')
parser.add_argument('--rank_batch_size', type = int, default = 64, help = 'batch size for ranker')
parser.add_argument('--ref_size', type = int, default = 16, help = 'Reference size in ranker')
parser.add_argument('--pre_g_epoch', type = int, default = 120, help = 'pretrain epochs for generator with MLE')
parser.add_argument('--pre_r_epoch', type = int, default = 50, help = 'pretrain epochs for rankder')
parser.add_argument('--pre_g_lr', type = float, default = 0.01, help = 'learning rate for generator MLE pretrain')
parser.add_argument('--ad_g_lr', type = float, default = 0.01, help = 'learning rate for generator MLE pretrain')
parser.add_argument('--rank_lr', type = float, default = 0.0001, help = 'learning rate for ranker')
parser.add_argument('--epoch', type = float, default = 200, help = 'training epoch for adversarial training.')
parser.add_argument('--g_step', type = int, default = 1, help = 'step for training generator in one epoch')
parser.add_argument('--r_step', type = int, default = 5, help = 'step for training ranker in one epoch')
parser.add_argument('--rollout_ratio', type = float, default = 0.8, help = 'Ratio for rollout model update')
parser.add_argument('--rollout_num', type = int, default = 16, help = 'rollout number')
parser.add_argument('--save_model', type = bool, default = False, help = "whether save model")
parser.add_argument('--restore_model', type = bool, default = False, help = "whether restore model")
parser.add_argument('--prefix', type = str, default = 'model', help = "prefix name for model save and log")
FLAGS = parser.parse_args()



def main():
    opt = Options()
    create_logging(FLAGS)

    random.seed(SEED)
    np.random.seed(SEED)
    # data loader
    gen_data_loader = Gen_Data_loader(FLAGS.gen_pre_batch_size)
    likelihood_data_loader = Gen_Data_loader(FLAGS.gen_pre_batch_size) # For testing
    rank_data_loader = Rank_Data_loader(FLAGS.rank_batch_size, FLAGS.ref_size)
    # network initialization
    generator = Generator(opt, FLAGS, pretrain = True)
    target_params = cPickle.load(open(opt.target_path))
    target_lstm = TARGET_LSTM(opt, FLAGS, target_params, pretrain = True) # The oracle model
    ranker = Ranker(opt, FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # create positive files for MLE training
    generate_samples(sess, target_lstm, FLAGS.gen_pre_batch_size, opt.generated_num, opt.positive_file)
    gen_data_loader.create_batches(opt.positive_file)

    #################################################################pretraining with MLE
     # pre-train generator
    logging.info('Start pre-training generator')
    for epoch in xrange(FLAGS.pre_g_epoch):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, FLAGS.gen_pre_batch_size, opt.generated_num, opt.eval_file)
            likelihood_data_loader.create_batches(opt.eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            logging.info("Pretrain generator epoch: %d, test_loss: %0.4f" % (epoch, test_loss))


    logging.info('Start pre-training rankder')
    # Train 3 epoch on the generated data and do this for 50 times
    for epoch in range(FLAGS.pre_r_epoch):
        generate_samples(sess, generator, FLAGS.gen_pre_batch_size, opt.generated_num, opt.negative_file)
        rank_data_loader.load_train_data(opt.positive_file, opt.negative_file)
        for _ in range(3):
            rank_data_loader.reset_pointer()
            for it in xrange(rank_data_loader.num_batch):
                x_batch, y_batch, ref = rank_data_loader.next_batch()
                feed = {
                    ranker.input_x: x_batch,
                    ranker.input_y: y_batch,
                    ranker.input_ref: ref,
                    ranker.dropout_keep_prob: opt.dropout_ratio
                }
                _, loss = sess.run([ranker.train_op, ranker.loss], feed)
        if epoch % 5 == 0:
            logging.info("Pretrain ranker epoch: %d, training loss: %0.4f" % (epoch, loss))

    
    # # # Save all params to disk.
    save_path = saver.save(sess, "./save/pre_model.ckpt")
    print("pretrain Model saved in file: %s" % save_path)
    
    # modify generator batch size for adversarial training
    tf.reset_default_graph()
    generator = Generator(opt, FLAGS, pretrain = False)
    ranker = Ranker(opt, FLAGS)
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(opt, FLAGS, target_params, pretrain = False) # The oracle model

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    # load parameters
    saver.restore(sess, "./save/pre_model.ckpt")
    likelihood_data_loader = Gen_Data_loader(FLAGS.gen_batch_size) # For testing
    print("Model restored.")
    rollout = ROLLOUT(generator, FLAGS.rollout_ratio, FLAGS.rollout_num)


    logging.info('#########################################################################')
    logging.info('Start adversarial training.')
    for epoch in range(FLAGS.epoch):
        # Train the generator for one step
        for it in range(FLAGS.g_step):
            samples = generator.generate(sess)
            generate_samples(sess, generator, FLAGS.gen_batch_size, opt.generated_num, opt.negative_file)
            rank_data_loader.load_train_data(opt.positive_file, opt.negative_file)
            rewards = rollout.get_reward(sess, samples, FLAGS.rollout_num, ranker, rank_data_loader)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Testing
        if epoch % 5 == 0 or epoch == epoch - 1:
            generate_samples(sess, generator, FLAGS.gen_batch_size, opt.generated_num, opt.eval_file)
            likelihood_data_loader.create_batches(opt.eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            logging.info("Epoch: %d, test_loss: %0.4f" % (epoch, test_loss))

        # Update roll-out parameters
        rollout.update_params()

        # Train the ranker
        for idx in range(FLAGS.r_step):
            generate_samples(sess, generator, FLAGS.gen_batch_size, opt.generated_num, opt.negative_file)
            rank_data_loader.load_train_data(opt.positive_file, opt.negative_file)

            for it in xrange(rank_data_loader.num_batch):
                x_batch, y_batch, ref = rank_data_loader.next_batch()
                feed = {
                    ranker.input_x: x_batch,
                    ranker.input_y: y_batch,
                    ranker.input_ref: ref,
                    ranker.dropout_keep_prob: opt.dropout_ratio
                }
                _ = sess.run(ranker.train_op, feed)


if __name__ == '__main__':
    main()
