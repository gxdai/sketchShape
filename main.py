import argparse
import os
import scipy.misc
import numpy as np
import os
# from model import model
from modelWeighted import model
#from model_ce import shapeCompletion            # The model is borrowed from context encoding
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gmSize', dest='gmSize', type=int, default=400, help='The size of ground metric')
parser.add_argument('--lamb', dest='lamb', type=float, default=100., help='parameter for sinkhorn iteration')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str, default='./checkpoint', help='directory for saving checkpoint')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='# images in batch')
parser.add_argument('--margin', type=float, default=10, help='The margin of contrastive loss')
parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file')
parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file')
parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The root direcoty of input data')
parser.add_argument('--num_views', type=int, default=20, help='The total number of views')
parser.add_argument('--num_views_sketch', type=int, default=20, help='The total number of views for sketches')
parser.add_argument('--num_views_shape', type=int, default=20, help='The total number of views for shape')
parser.add_argument('--class_num', type=int, default=90, help='the total number of class')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, evaluation')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='name of the dataset')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=10000, help='maximum number of iterations')
parser.add_argument('--inputFeaSize', dest='inputFeaSize', type=int, default=4096, help='The dimensions of input features')
parser.add_argument('--outputFeaSize', dest='outputFeaSize', type=int, default=100, help='The dimensions of input features')
parser.add_argument('--lossType', dest='lossType', type=str, default='contrastiveLoss', help='name of the dataset')
parser.add_argument('--activationType', dest='activationType', type=str, default='relu', help='name of the dataset')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default='0.001', help='learning rate')
parser.add_argument('--normFlag', dest='normFlag', type=int, default=0, help='learning rate')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.5, help='momentum term of Gradient')
parser.add_argument('--weightFile', dest='weightFile', type=str, default='tmp', help='Weight file for evaluation')
parser.add_argument('--returnDir', dest='returnDir', type=str, default='returnDir', help='Weight file for evaluation')

"""
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='name of the dataset')
parser.add_argument('--imageRootDir', dest='imageRootDir', default=None, help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--output_size', dest='output_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--shape_size', dest='shape_size', type=int, default=32, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--num_gpus', dest='num_gpus', type=int, default=3, help='# of gpus')

parser.add_argument('--train_listFile', dest='train_listFile', default='./data/h5_shapenet_dim32_sdf/train_file_label.txt', help='training list file')
parser.add_argument('--test_benchmark', dest='test_benchmark', default='./benchmark_test.txt', help='training list file')
parser.add_argument('--benchmark_output_dir', dest='benchmark_output_dir', default='./evaluation', help='training list file')
parser.add_argument('--test_listFile', dest='test_listFile', default='./data/h5_shapenet_dim32_sdf/test_file_label.txt', help='testing list file')
parser.add_argument('--fileRootDir', dest='fileRootDir', default='/home/gxdai/MMVC_LARGE/Guoxian_Dai/data/shapeCompletion/txt', help='testing list file')
parser.add_argument('--testInputType', dest='testInputType', default='shapenet_dim32_sdf', help='testing list file')
parser.add_argument('--truncation', type=float, default=3, help='The truncation threshold of input voxel')

# The dropoutCondtion for generator (with dropout or without dropout)
parser.add_argument('--dropoutCondition', type=int, default=1, help='Whether to use dropout in generator')
"""
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(args.ckpt_dir)
    wasserteinModel = model(gmSize=args.gmSize, lamb=args.lamb, ckpt_dir=args.ckpt_dir, batch_size=args.batch_size, margin=args.margin, weightFile=args.weightFile,
            sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list, num_views=args.num_views,returnDir=args.returnDir,
            num_views_sketch=args.num_views_sketch, num_views_shape=args.num_views_shape, learning_rate=args.learning_rate, momentum=args.momentum,
            class_num=args.class_num, normFlag=args.normFlag, logdir=args.logdir, lossType=args.lossType, activationType=args.activationType,
            phase=args.phase, inputFeaSize=args.inputFeaSize, outputFeaSize=args.outputFeaSize, maxiter=args.maxiter)


    if args.phase == 'train':
        wasserteinModel.train()
    elif args.phase == 'evaluation':
        print("evaluating using weighted feature")
        wasserteinModel.evaluation()

if __name__ == '__main__':
    tf.app.run()
