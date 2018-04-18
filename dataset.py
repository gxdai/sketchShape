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
        self.sketch_train_size = len(lines) / self.num_views
        self.sketch_train_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]

        # Testing data
        with open(sketch_test_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines] 
        self.sketch_test_size = len(lines) / self.num_views
        self.sketch_test_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]
        self.sketch_test_label = [int(line[0].split(' ')[-1]) for line in self.sketch_test_data]     # The label of testing sketch

        # Shape data
        with open(shape_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.shape_size = len(lines) / self.num_views
        self.shape_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]
        self.shape_label = [int(line[0].split(' ')[-1]) for line in self.shape_data]     # The label of shape


        self.groupByClass()
        self.shuffleList()
        
        self.train_ptr = 0
        self.test_ptr = 0
    
    def groupByClass(self):
        # init the dictionary for different classes
        self.shape = {}
        self.sketch_train = {}
        self.sketch_test = {}

        for i in range(self.class_num):     # initiate all class as empty list
            self.shape[str(i)] ={}
            self.shape[str(i)]['data'] = []
            self.sketch_train[str(i)] = {}
            self.sketch_train[str(i)]['data'] = []
            self.sketch_test[str(i)] = {}
            self.sketch_test[str(i)]['data'] = [] 

        # training sketch grouped by class
        for line in self.sketch_train_data:
            self.sketch_train[line[0].split(' ')[-1]]['data'].append(line)
        # testing sketch grouped by class
        for line in self.sketch_test_data:
            self.sketch_test[line[0].split(' ')[-1]]['data'].append(line)
        # Shape grouped by class
        for line in self.shape_data:
            self.shape[line[0].split(' ')[-1]]['data'].append(line)

        # update the total number of samples for each class
        for i in range(self.class_num):
            self.shape[str(i)]['total_num'] = len(self.shape[str(i)]['data'])
            self.sketch_train[str(i)]['total_num'] = len(self.sketch_train[str(i)]['data'])
            self.sketch_test[str(i)]['total_num'] = len(self.sketch_test[str(i)]['data'])
        """
        print(self.sketch_train.keys())
        print(len(self.sketch_train.keys()))
        print(self.sketch_train['0']['data'][0])
        print(self.sketch_train['0']['data'][1])
        print(self.sketch_test.keys())
        print(len(self.sketch_test.keys()))
        print(self.sketch_test['0']['data'][0])
        print(self.sketch_test['0']['data'][1])
        print(self.shape.keys())
        print(len(self.shape.keys()))
        print(self.shape['0']['data'][0])
        print(self.shape['0']['data'][1])
        """

    def shuffleList(self):
        # Construct pair of sketch and shape data
        self.train_pair = []
        self.test_pair  = []
        # Get the pairwise samples for each class
        for i in range(self.class_num):
            # shuffle the samples for each class
            # for each class, Get the maximium number of sketch and shape 
            class_i_train_number = max(self.shape[str(i)]['total_num'], self.sketch_train[str(i)]['total_num'])
            class_i_test_number = max(self.shape[str(i)]['total_num'], self.sketch_test[str(i)]['total_num'])


            # shuffle training data 
            rand_index_shape = np.random.permutation(class_i_train_number) % self.shape[str(i)]['total_num']
            rand_index_sketch = np.random.permutation(class_i_train_number) % self.sketch_train[str(i)]['total_num']
            for j in range(class_i_train_number):
                self.train_pair.append(zip(self.sketch_train[str(i)]['data'][rand_index_sketch[j]], self.shape[str(i)]['data'][rand_index_shape[j]])) 

            # shuffle testing data
            rand_index_shape = np.random.permutation(class_i_test_number) % self.shape[str(i)]['total_num']
            rand_index_sketch = np.random.permutation(class_i_test_number) % self.sketch_test[str(i)]['total_num']
            for j in range(class_i_test_number):
                self.test_pair.append(zip(self.sketch_test[str(i)]['data'][rand_index_sketch[j]], self.shape[str(i)]['data'][rand_index_shape[j]])) 

        shuffle(self.train_pair)
        shuffle(self.test_pair)
        self.train_pair_number = len(self.train_pair) 
        self.test_pair_number = len(self.test_pair) 



        """
        ### Load test sketch ##############3
        with open(sketch_test_list) as f:
            lines = f.readlines()
            ### only pick 5 groups for training####
            lines = lines[:150]
            self.sketch_test_fea = []
            self.sketch_test_label = []
            shuffle(lines)
            for l in lines:
                items = l.split()
                self.sketch_test_fea.append(items[0])
                self.sketch_test_label.append(items[1])
        with open(shape_list) as f:
            lines = f.readlines()
            #### only pick g groups ####
            lines = lines[:202]
            self.shape_fea = []
            self.shape_label = []
            shuffle(lines)
            for l in lines:
                items = l.split()
                self.shape_fea.append(items[0])
                self.shape_label.append(items[1])
        self.sketch_train_ptr = 0       ## pointer training sketch
        self.sketch_test_ptr = 0       ## pointer for testing sketch
        self.shape_ptr = 0
        self.sketch_train_size = len(self.sketch_train_label)
        self.sketch_test_size = len(self.sketch_test_label)
        self.shape_size = len(self.shape_label)
        ### whether shuffle data at the start of each epoch
        self.shuffle = 1
        """

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_pair_number:
                paths = self.train_pair[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                # shuffle the list
                self.shuffleList()
                self.train_ptr = 0
                paths = self.train_pair[self.train_ptr: self.train_ptr+batch_size]
                self.train_ptr += batch_size
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_pair_number:
                paths = self.test_pair[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                self.shuffleList()
                self.test_ptr = 0
                paths = self.test_pair[self.test_ptr: self.test_ptr + batch_size]
                self.test_ptr += batch_size
        else:
            return None, None

        # Read images
        sketch_fea = np.ndarray([batch_size, self.num_views, 4096])     ### 4096 is the feature size
        shape_fea = np.ndarray([batch_size, self.num_views, 4096])     ### 4096 is the feature size
        for i, path in enumerate(paths):
            for j in range(self.num_views):
                sketch_file = paths[i][j][0].split(' ')[0]
                shape_file = paths[i][j][1].split(' ')[0]
                sketch_fea[i,j] = np.loadtxt(sketch_file)
                shape_fea[i,j] = np.loadtxt(shape_file)
        return sketch_fea, shape_fea

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
        sketch_test_feaset = np.zeros((self.sketch_test_size, 4096))
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
        sketch_train_feaset = np.zeros((self.sketch_train_size, 4096))
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
        shape_feaset = np.zeros((self.shape_size, 4096))
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
"""
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
    print(data.train_pair_number)
    for _ in range(10000):
        start_time = time.time()
        sketch_fea, shape_fea = data.next_batch(5, 'train')
        print("time cost: {}".format(time.time() - start_time))
        print(sketch_fea[:,:,:2])
        print("#############################\n")
        print(shape_fea[:,:,:2])
        print("#############################\n")
"""
