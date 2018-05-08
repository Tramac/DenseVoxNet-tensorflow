from __future__ import print_function

import argparse
import os
import time

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from BatchDataReader import BatchDataset
from model_v2 import DenseVoxNet
from utils import pre_process_isotropic, generate_score_map_partition, generate_score_map_patch2Img, partition_Img, \
    patches2Img_vote, imresize3d, RemoveMinorCC

parser = argparse.ArgumentParser(description="train VoxDenseNet")
parser.add_argument("--iteration", "-i", default=15000, type=int, help="number of iterations, default=15000")
parser.add_argument("--dispaly_step", "-s", default=1000, type=int, help="number of steps to display, default=1000")
parser.add_argument("--input_file", "-f", default="train.txt", type=str, help="file of training dataset")
parser.add_argument("--random_crop", default=True, type=bool, help="random crop to images")
parser.add_argument("--batch_size", default=1, type=int, help="batch size, default=3")
parser.add_argument("--input_size", default=64, type=int, help="shape of input for the network, default=64")
parser.add_argument("--learning_rate", "-r", default=0.05, type=float, help="learning rate, default=0.05")
parser.add_argument("--logs_dir", default="logs/", type=str, help="location of trained model")
parser.add_argument("--data_zoo", default="data/", type=str, help="location of data")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mode", default="test", type=str, help="train/test")
args = parser.parse_args()

train_params = vars(args)

print("Params:")
for k, v in train_params.items():
    print("%s: %s" % (k, v))

print("Initialize the model...")
if args.mode == "train":
    is_training = True
else:
    is_training = False
net = DenseVoxNet(is_training=is_training)


def main(argv=None):
    shape = [None, args.input_size, args.input_size, args.input_size, 1]
    image = tf.placeholder(tf.float32, shape=shape, name="input_image")
    label = tf.placeholder(tf.int32, shape=shape, name="label")

    logits1, logits2, prob_map, pred_annotation = net(image)
    loss1 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=tf.squeeze(label, squeeze_dims=[4]),
                                                       name="entropy_1"))

    loss2 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=tf.squeeze(label, squeeze_dims=[4]),
                                                       name="entropy_2"))
    loss = loss1 + 0.33 * loss2
    tf.summary.scalar("entropy", loss)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.polynomial_decay(args.learning_rate, global_step, args.iteration, power=0.9)
    tf.summary.scalar("learning_rate", learning_rate)

    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    with open(args.input_file) as f:
        train_dataset_list = f.read().splitlines()

    print("Setting up dataset reader...")
    train_dataset_reader = BatchDataset(train_dataset_list)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(args.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if args.mode == "train":
        for itr in xrange(args.iteration + 1):
            start_time = time.time()
            train_images, train_labels = train_dataset_reader.next_batch(args.batch_size, args.random_crop,
                                                                         args.input_size)
            feed_dict = {image: train_images, label: train_labels}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 1 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                end_time = time.time()
                print("Step: %d, Train loss: %g, Time: %gs" % (itr, train_loss, end_time - start_time))
                summary_writer.add_summary(summary_str, itr)

            if itr % 10 == 0:
                """valid"""
                saver.save(sess, args.logs_dir + "model.ckpt", itr)

    if args.mode == "test":
        # parameters
        use_isotropic = 0
        ita = 4
        tr_mean = 0

        for id in range(0, 1):
            # read data and pre-processing
            tic = time.time()
            print("test sample #%d\n" % id)
            pred_list = []
            vol_path = os.path.join(args.data_zoo, "training_axial_crop_pat" + str(id) + ".nii.gz")
            vol_src = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))
            vol_src = np.transpose(vol_src, [0, 2, 1])
            vol_data, _ = pre_process_isotropic(vol_src, [], use_isotropic, id)
            data = vol_data - tr_mean

            vol_label = os.path.join(args.data_zoo, "training_axial_crop_pat" + str(id) + "-label.nii.gz")
            vol_label = sitk.GetArrayFromImage(sitk.ReadImage(vol_label))
            vol_label = np.transpose(vol_label, [0, 2, 1])
            vol_label, _ = pre_process_isotropic(vol_label, [], use_isotropic, id)

            # average fusion scheme
            patch_list_avg, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = generate_score_map_partition(
                data, args.input_size, ita)

            patch_list_label, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = generate_score_map_partition(
                vol_label, args.input_size, ita)

            res_list_avg = []
            for i in range(len(patch_list_avg)):
                crop_data = patch_list_avg[i]
                crop_label = patch_list_label[i]

                crop_data = np.expand_dims(np.expand_dims(crop_data, axis=0), axis=4)
                crop_label = np.zeros((1, args.input_size, args.input_size, args.input_size, 1))
                res_score = sess.run(prob_map, feed_dict={image: crop_data, label: crop_label})
                res_score = np.squeeze(res_score)

                res_list_avg.append(res_score)
            score_map = generate_score_map_patch2Img(res_list_avg, ss_h, ss_w, ss_l, padding_size_x, padding_size_y,
                                                     padding_size_z, args.input_size, ita)
            avg_label = np.argmax(score_map, axis=3)

            # major voting scheme
            patch_list_vote, r, c, h = partition_Img(data, args.input_size, ita)
            for i in range(len(patch_list_vote)):
                crop_data = patch_list_vote[i]
                crop_data = np.expand_dims(np.expand_dims(crop_data, axis=0), axis=4)
                crop_label = np.zeros((1, args.input_size, args.input_size, args.input_size, 1))
                res_L2 = sess.run(prob_map, feed_dict={image: crop_data, label: crop_label})
                res_L2 = np.squeeze(res_L2)
                res_lable = np.argmax(res_L2, axis=3)
                pred_list.append(res_lable)
            fusion_Img_L2, vote_label = patches2Img_vote(pred_list, r, c, h, args.input_size, ita)

            # post-processing
            if use_isotropic == 1:
                vote_label = imresize3d(vote_label, [], vol_src.shape, 'reflect')
                avg_label = imresize3d(avg_label, [], vol_src.shape, 'reflect')

            # remove minor connected components
            vote_label = RemoveMinorCC(vote_label, 0.2)
            avg_label = RemoveMinorCC(avg_label, 0.2)


if __name__ == "__main__":
    tf.app.run()
