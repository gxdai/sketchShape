import numpy as np
from random import shuffle
import scipy.io as sio
import argparse
import time
import sys
class Dataset:

    def __init__(self, sketch_train_list, sketch_test_list, shape_list, num_views=20, class_num=90):
        # Load training images (path) and labels
        """
        num_views:      The number of views 
        class_num:      The total number of class
        """ 
        self.num_views = num_views
        self.class_num = class_num

        # Training data
        with open(sketch_train_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.sketch_train_num = len(lines) / self.num_views
        self.sketch_train_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]

        # Shuffle sketch train data
        shuffle(self.sketch_train_data)

        # Testing data
        with open(sketch_test_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines] 
        self.sketch_test_num = len(lines) / self.num_views
        self.sketch_test_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]

        # Shape data
        with open(shape_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.shape_num = len(lines) / self.num_views
        self.shape_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]
        shuffle(self.shape_data)
        
        # set up the pointer
        self.sketch_train_ptr = 0
        self.sketch_test_ptr = 0
        self.shape_ptr = 0

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            # Load training sketch
            if self.sketch_train_ptr + batch_size < self.sketch_train_num:
                sketch_paths = self.sketch_train_data[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]
                self.sketch_train_ptr += batch_size
            else:
                # shuffle the list
                shuffle(self.sketch_train_data)
                self.sketch_train_ptr = 0
                sketch_paths = self.sketch_train_data[self.sketch_train_ptr: self.sketch_train_ptr+batch_size]
                self.sketch_train_ptr += batch_size
            # Loading training shapes
            if self.shape_ptr + batch_size < self.shape_num:
                shape_paths = self.shape_data[self.shape_ptr:self.shape_ptr + batch_size]
                self.shape_ptr += batch_size
            else:
                # shuffle the list
                shuffle(self.shape_data)
                self.shape_ptr = 0
                shape_paths = self.shape_data[self.shape_ptr: self.shape_ptr+batch_size]
                self.shape_ptr += batch_size
            pairPaths = zip(sketch_paths, shape_paths)
        #TO BE CONTINUED
       
        # Read images
        sketch_fea = np.zeros((batch_size, 1, 4096))     ### 4096 is the feature size
        sketch_label = np.zeros((batch_size, 1))     ### 4096 is the feature size
        shape_fea = np.zeros((batch_size, self.num_views, 4096))     ### 4096 is the feature size
        shape_label = np.zeros((batch_size, 1))     ### 4096 is the feature size
        for i, paths in enumerate(pairPaths):
            # only one sketch is needed 
            sketch_file = paths[0][0].split(' ')
            sketch_fea[i,0] = np.loadtxt(sketch_file[0])
            sketch_label[i,0] = int(sketch_file[1])
            for j in range(self.num_views):
                shape_file = paths[1][j].split(' ')
                shape_fea[i,j] = np.loadtxt(shape_file[0])
                shape_label[i,0] = int(shape_file[1])

        return sketch_fea, sketch_label, shape_fea, shape_label


    def getLabel(self):
        # Get sketch test label
        self.sketch_test_label = []
        for tmp_sketch in self.sketch_test_data:
            self.sketch_test_label.append(int(tmp_sketch[0].split(' ')[-1]))

        # Get shape label
        self.shape_label = []
        for tmp_shape in self.shape_data:
            self.shape_label.append(int(tmp_shape[0].split(' ')[-1]))

        
    def retrievalParamSP(self):
        shapeLabels = np.array(self.shape_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths
    def retrievalParamSS(self):
        shapeLabels = np.array(self.sketch_train_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths
    def retrievalParamPP(self):
        shapeLabels = np.array(self.shape_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.shape_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths


    def normalizeData(self):
        ########### normalize sketch test feature ######################
        #print('Processing testing sketch\n')
        sketch_test_feaset = np.zeros((self.sketch_test_num, 4096))
        for i in range(len(self.sketch_test_label)):
            # print(i)
            mat_contents = sio.loadmat(self.sketch_test_fea[i])
            img = mat_contents['fea']
            #print(img)
            sketch_test_feaset[i] = img
        sketch_test_mean = np.mean(sketch_test_feaset, axis=0)
        sketch_test_std = np.std(sketch_test_feaset, axis=0)
        sketch_test_feaset_norm = (sketch_test_feaset - sketch_test_mean) / sketch_test_std
        #print(np.where(np.isnan(sketch_test_feaset_norm)))
        ########### nomralize sketch train feature #####################
        #print('Processing training sketch')
        sketch_train_feaset = np.zeros((self.sketch_train_num, 4096))
        for i in range(len(self.sketch_train_label)):
            # print(i)
            mat_contents = sio.loadmat(self.sketch_train_fea[i])
            img = mat_contents['fea']
            #print(img)
            sketch_train_feaset[i] = img
        sketch_train_mean = np.mean(sketch_train_feaset, axis=0)
        sketch_train_std = np.std(sketch_train_feaset, axis=0)
        sketch_train_feaset_norm = (sketch_train_feaset - sketch_train_mean) / sketch_train_std
        #print(np.where(np.isnan(sketch_train_feaset_norm)))
        ########## normalize shape feature ###################
        #print('Processing shape\n')
        shape_feaset = np.zeros((self.shape_num, 4096))
        for i in range(len(self.shape_label)):
            # print(i)
            mat_contents = sio.loadmat(self.shape_fea[i])
            img = mat_contents['coeff']
            shape_feaset[i] = img
        shape_mean = np.mean(shape_feaset, axis=0)
        shape_std = np.std(shape_feaset, axis=0)
        shape_feaset_norm = (shape_feaset - shape_mean) / shape_std
        ###### get rid of nan Dataset################
        shape_feaset_norm[np.where(np.isnan(shape_feaset_norm))] = 0
        #print(np.where(np.isnan(shape_feaset_norm)))
        return sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This is for loading shapenet partial data')
    parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file')
    parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file')
    parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The root direcoty of input data')
    parser.add_argument('--num_views', type=int, default=20, help='The total number of views')
    parser.add_argument('--class_num', type=int, default=40, help='the total number of class')
    args = parser.parse_args() 
    data = Dataset(sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list,
            num_views=args.num_views, class_num=args.class_num)
    print('\n\n\n\n\n\n\n')
    for _ in range(10000):
        start_time = time.time()
        sketch_fea, sketch_label, shape_fea, shape_label = data.next_batch(5, 'train')
        print("time cost: {}".format(time.time() - start_time))
        print(sketch_fea[:,:,:2])
        print("#############################\n")
        print(shape_fea[:,:,:2])
        print("#############################\n")

