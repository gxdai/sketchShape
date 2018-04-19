# import the required modules
import argparse
import os
import scipy.misc
import numpy as np
from model import model
import tensorflow as tf

# define the arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str, default='./checkpoint', help='directory for saving checkpoint')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='batch size')
parser.add_argument('--margin', type=float, default=10, help='The margin of contrastive loss')
parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file for sketch')
parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file for sketch')
parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The list file for shape')
parser.add_argument('--num_views_shape', type=int, default=12, help='The total number of rendered views for shape')
parser.add_argument('--phase', dest='phase', default='train', help='train or evaluation')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='The directory for training logs')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=10000, help='maximum number of iterations')
parser.add_argument('--inputFeaSize', dest='inputFeaSize', type=int, default=4096, help='The dimensions of input features')
parser.add_argument('--outputFeaSize', dest='outputFeaSize', type=int, default=100, help='The dimensions of output features')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default='0.001', help='learning rate')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.5, help='momentum term of Gradient')
parser.add_argument('--weightFile', dest='weightFile', type=str, default='model-2000.ckpt', help='Weight file for evaluation')
# end of argument definition
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(args.ckpt_dir)
    weightedModel = model(ckpt_dir=args.ckpt_dir, batch_size=args.batch_size, margin=args.margin, weightFile=args.weightFile,
            sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list,
            num_views_shape=args.num_views_shape, learning_rate=args.learning_rate, momentum=args.momentum,
            logdir=args.logdir, phase=args.phase, inputFeaSize=args.inputFeaSize, outputFeaSize=args.outputFeaSize, maxiter=args.maxiter)
    if args.phase == 'train':
        weightedModel.train()
    elif args.phase == 'evaluation':
        print("evaluating using weighted feature")
        weightedModel.evaluation()

if __name__ == '__main__':
    tf.app.run()
