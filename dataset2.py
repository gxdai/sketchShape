import numpy as np
from random import shuffle
import scipy.io as sio
import argparse
import time
import sys
class Dataset:
    def __init__(self, sketch_train_list, sketch_test_list, shape_list, num_views=20, num_views_sketch=20, num_views_shape=20, feaSize=4096, class_num=90, phase='train', normFlag=0):
        # Load training images (path) and labels
        """
        num_views:      The number of views 
        class_num:      The total number of class
        """ 
        self.num_views = num_views
        self.class_num = class_num
        self.num_views_sketch = num_views_sketch
        self.feaSize = feaSize
        self.phase = phase
        self.normFlag = normFlag

        # Training data
        with open(sketch_train_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.sketch_train_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]
        self.sketch_train_num = len(self.sketch_train_data)
        # Shuffle sketch train data
        shuffle(self.sketch_train_data)

        # Testing data
        with open(sketch_test_list) as f:
            lines = f.readlines() 
        lines = [line.rstrip('\n') for line in lines] 
        self.sketch_test_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)] 
        self.sketch_test_num = len(self.sketch_test_data)

        # Shape data
        with open(shape_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.shape_data = [lines[i:i+self.num_views] for i in range(0, len(lines), self.num_views)]
        shuffle(self.shape_data)
        self.shape_num = len(self.shape_data)

        # all the pointer 
        self.sketch_train_ptr = 0 
        self.sketch_test_ptr = 0 
        self.shape_ptr = 0 




        # Load all the data
        self.loadAllData(phase)

        if self.normFlag:
            self.normalizeData(phase)

        # create random index for all data
        self.sketch_test_randIndex = np.random.permutation(self.sketch_test_num)
        self.sketch_train_randIndex = np.random.permutation(self.sketch_train_num)
        self.shape_randIndex = np.random.permutation(self.shape_num)



    def loadAllData(self, phase):
        def loadFeaAndLabel(pathSet, num_views, feaSize):
            sampleNum = len(pathSet)
            feaSet = np.zeros((sampleNum, num_views, feaSize))
            labelSet = np.zeros((sampleNum, 1))
            for i in range(sampleNum):
                for k in range(num_views):
                    filePath = pathSet[i][k].split(' ')
                    feaSet[i,k] = self.loaddata(filePath[0])
                    labelSet[i] = int(filePath[1])

            return feaSet, labelSet
        if phase == 'evaluation':
            print("Load sketch testing features")
            start_time = time.time()
            self.sketchTestFeaset, self.sketchTestLabelset = loadFeaAndLabel(self.sketch_test_data, self.num_views_sketch, self.feaSize)
            print("Loading time: {}".format(time.time() - start_time))

        elif phase == 'train': 
            print("Load sketch training features")
            start_time = time.time()
            self.sketchTrainFeaset, self.sketchTrainLabelset = loadFeaAndLabel(self.sketch_train_data, self.num_views_sketch, self.feaSize)
            print("Loading time: {}".format(time.time() - start_time))
        
        print("Load shape features")
        start_time = time.time()
        self.shapeFeaset, self.shapeLabelset = loadFeaAndLabel(self.shape_data, self.num_views_shape, self.feaSize)
        print("Loading time: {}".format(time.time() - start_time))
        print("Finish Loading")

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            # Load training sketch
            if self.sketch_train_ptr + batch_size < self.sketch_train_num:
                sketch_paths = self.sketch_train_data[self.sketch_train_ptr:self.sketch_train_ptr+batch_size]
                self.sketch_train_ptr += batch_size
            else:
                # shuffle the list
                shuffle(self.sketch_train_data)
                self.sketch_train_ptr = 0
                sketch_paths = self.sketch_train_data[self.sketch_train_ptr:self.sketch_train_ptr+batch_size]
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
        sketch_fea = np.zeros((batch_size, self.num_views_sketch, 4096))     ### 4096 is the feature size
        sketch_label = np.zeros((batch_size, 1))     ### 4096 is the feature size
        shape_fea = np.zeros((batch_size, self.num_views_shape, 4096))     ### 4096 is the feature size
        shape_label = np.zeros((batch_size, 1))     ### 4096 is the feature size
        for i, paths in enumerate(pairPaths):
            for j in range(self.num_views_sketch):
                sketch_file = paths[0][j].split(' ')
                #sketch_fea[i,j] = np.loadtxt(sketch_file[0])
                sketch_fea[i,j] = self.loaddata(sketch_file[0])
                sketch_label[i,0] = int(sketch_file[1])
            for j in range(self.num_views_shape):
                shape_file = paths[1][j].split(' ')
                #shape_fea[i,j] = np.loadtxt(shape_file[0])
                shape_fea[i,j] = self.loaddata(shape_file[0])
                shape_label[i,0] = int(shape_file[1])

        return sketch_fea, sketch_label, shape_fea, shape_label

    def nextBatch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'sketch_train':
            # Load training sketch
            if self.sketch_train_ptr + batch_size <= self.sketch_train_num:
                batchIndex = self.sketch_train_randIndex[self.sketch_train_ptr:self.sketch_train_ptr+batch_size]
                sketch_fea = self.sketchTrainFeaset[batchIndex]
                sketch_label = self.sketchTrainLabelset[batchIndex]
                self.sketch_train_ptr += batch_size
            else:
                # shuffle the list
                self.sketch_train_randIndex = np.random.permutation(self.sketch_train_num)
                self.sketch_train_ptr = 0
                batchIndex = self.sketch_train_randIndex[self.sketch_train_ptr:self.sketch_train_ptr+batch_size]
                sketch_fea = self.sketchTrainFeaset[batchIndex]
                sketch_label = self.sketchTrainLabelset[batchIndex]
                self.sketch_train_ptr += batch_size

            return sketch_fea, sketch_label
        elif phase == 'shape':
            # Loading training shapes
            if self.shape_ptr + batch_size <= self.shape_num:
                batchIndex = self.shape_randIndex[self.shape_ptr:self.shape_ptr+batch_size]
                shape_fea = self.shapeFeaset[batchIndex]
                shape_label = self.shapeLabelset[batchIndex]
                self.shape_ptr += batch_size
            else:
                # shuffle the list
                self.shape_randIndex = np.random.permutation(self.shape_num)
                self.shape_ptr = 0
                batchIndex = self.shape_randIndex[self.shape_ptr:self.shape_ptr+batch_size]
                shape_fea = self.shapeFeaset[batchIndex]
                shape_label = self.shapeLabelset[batchIndex]
                self.shape_ptr += batch_size

            return shape_fea, shape_label



    def loaddata(self, filepath):
        fid = open(filepath, 'r')
        lines = fid.readlines()
        return np.array(lines).astype(float)


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
        for i in range(unique_labels.shape[0]):             ### find the numbers tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
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


    def normalizeData(self, phase):
        ########### normalize sketch test feature ######################
        #print('Processing testing sketch\n')
        print("Normalizing shape features")


        shape_mean = np.mean(self.shapeFeaset, axis=0)
        shape_std = np.std(self.shapeFeaset, axis=0)
        self.shapeFeaset = (self.shapeFeaset - shape_mean) / shape_std
        ###### get rid of nan Dataset################
        self.shapeFeaset[np.where(np.isnan(self.shapeFeaset))] = 0
        #print(np.where(np.isnan(shape_feaset_norm)))
       

       
        if self.phase == 'train':
            print("Normalizing sketch train features")
            sketch_train_mean = np.mean(self.sketchTrainFeaset, axis=0)
            sketch_train_std = np.std(self.sketchTrainFeaset, axis=0)
            self.sketchTrainFeaset = (self.sketchTrainFeaset - sketch_train_mean) / sketch_train_std
        elif self.phase == 'evaluation':
            print("Normalizing sketch test features")
            sketch_test_mean = np.mean(self.sketchTestFeaset, axis=0)
            sketch_test_std = np.std(self.sketchTestFeaset, axis=0)
            self.sketchTestFeaset = (self.sketchTestFeaset - sketch_test_mean) / sketch_test_std
        
        
      
       
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This is for loading shapenet partial data')
    parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file')
    parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file')
    parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The root direcoty of input data')
    parser.add_argument('--num_views', type=int, default=20, help='The total number of views')
    parser.add_argument('--num_views_sketch', type=int, default=1, help='The total number of views')
    parser.add_argument('--class_num', type=int, default=40, help='the total number of class')
    args = parser.parse_args() 
    data = Dataset(sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list,
            num_views=args.num_views, num_views_sketch=args.num_views_sketch, class_num=args.class_num)
    print('\n\n\n\n\n\n\n')
    for _ in range(1500):
        start_time = time.time()
        sketch_fea, sketch_label, shape_fea, shape_label = data.next_batch(5, 'train')
        print("time cost: {}".format(time.time() - start_time))
        print(sketch_fea.shape)
        print(sketch_label.shape)
        print("#############################\n")
        print(shape_fea.shape)
        print(shape_label.shape)
        print("#############################\n")
"""
