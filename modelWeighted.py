import tensorflow as tf
import numpy as np
import numpy.matlib
from scipy.spatial import distance
import os
import sys
# from dataset2 import Dataset
from datasetWeighted import Dataset
import time
from RetrievalEvaluation import RetrievalEvaluation
#tf.set_random_seed(222)
#np.random.seed(222)

class model(Dataset):
    """ Create the model for non-linear wasserstein metric """
    def __init__(self, gmSize=400, lamb=20., ckpt_dir='./checkpoint', ckpt_name='model', mapFile='map.txt',
            batch_size=30, margin=10., learning_rate=0.001, momentum=0.9, sketch_train_list=None,weightFile=None,returnDir=None,
            sketch_test_list=None, shape_list=None, num_views=20, num_views_sketch=20, num_views_shape=20, class_num=90, normFlag=0,
            logdir=None, lossType='contrastiveLoss', activationType='relu', phase='train', inputFeaSize=4096, outputFeaSize=100, maxiter=100000):
        """
        gmSize:         The size of ground metric matrix
        lamb:           The parameter for sinkhorn iteration
        ckpt_dir:       The directory for saving checkpoint file
        ckpt_name:      The name of checkpoint file
        batch_size:     The training batch_size
        """
        self.gmSize             =       gmSize                          # The size of the ground metric matrix: n**2
        self.lamb               =       lamb
        self.gmWidth            =       int(np.sqrt(gmSize))            # The width and height of ground metric matrix
        self.ckpt_dir           =       ckpt_dir
        self.ckpt_name          =       ckpt_name
        self.batch_size         =       batch_size
        self.logdir             =       logdir
        self.class_num          =       class_num
        self.num_views          =       num_views
        self.num_views_sketch   =       num_views_sketch
        self.num_views_shape    =       num_views_shape
        self.maxiter            =       maxiter
        self.inputFeaSize       =       inputFeaSize
        self.outputFeaSize      =       outputFeaSize
        self.margin             =       margin
        self.learning_rate      =       learning_rate
        self.momentum           =       momentum
        self.lossType           =       lossType
        self.phase              =       phase
        self.normFlag           =       normFlag
        self.activationType     =       activationType
        self.mapFile            =       mapFile
        self.weightFile         =       weightFile
        self.returnDir          =       returnDir

        print("self.gmSize             =       {:5d}".format(self.gmSize))
        print("self.lamb               =       {:2.5f}".format(self.lamb))
        print("self.gmWidth            =       {:5d}".format(self.gmWidth))
        print("self.ckpt_dir           =       {:10}".format(self.ckpt_dir))
        print("self.ckpt_name          =       {:10}".format(self.ckpt_name))
        print("self.batch_size         =       {:5d}".format(self.batch_size))
        print("self.logdir             =       {:10}".format(self.logdir))
        print("self.class_num          =       {:5d}".format(self.class_num))
        print("self.num_views          =       {:5d}".format(self.num_views))
        print("self.num_views_sketch   =       {:5d}".format(self.num_views_sketch))
        print("self.num_views_shape    =       {:5d}".format(self.num_views_shape))
        print("self.maxiter            =       {:5d}".format(self.maxiter))
        print("self.inputFeaSize       =       {:5d}".format(self.inputFeaSize))
        print("self.outputFeaSize      =       {:5d}".format(self.outputFeaSize))
        print("self.margin             =       {:2.5f}".format(self.margin))
        print("self.learning_rate      =       {:2.5f}".format(self.learning_rate))
        print("self.momentum           =       {:2.5f}".format(self.momentum))
        print("self.lossType           =       {:10}".format(self.lossType))
        print("self.phase              =       {:10}".format(self.phase))
        print("self.normFlag           =       {:10d}".format(self.normFlag))
        print("self.activationType     =       {:10}".format(self.activationType))


        Dataset.__init__(self,sketch_train_list=sketch_train_list, sketch_test_list=sketch_test_list, shape_list=shape_list, feaSize=inputFeaSize, class_num=class_num, phase=phase, normFlag=normFlag)
        self.build_model()

    def sketchNetwork(self, x):          #### for sketch network
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", 0.1)
            fc4 = self.fc_layer(fc3, self.outputFeaSize, "fc4", 0.1)

            return fc4, fc3
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.sigmoid(fc2)
            fc3 = self.fc_layer(ac2, self.outputFeaSize, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)

            return ac3, ac2

    def shapeNetwork(self, x):          #### for sketch network
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 2000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 1000, "fc3", stddev)
            ac3 = tf.nn.relu(fc3)
            fc4 = self.fc_layer(ac3, 500, "fc4", 0.1)
            fc5 = self.fc_layer(fc4, self.outputFeaSize, "fc5", 0.1)

            return fc5, fc4
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)
            fc4 = self.fc_layer(ac3, self.outputFeaSize, "fc4", stddev)
            ac4 = tf.nn.sigmoid(fc4)

            return ac4, ac3




    def weightNet(self, x):
        stddev=0.01
        fc1 = self.fc_layer(x, self.num_views_shape, "classify", stddev)
        prob = tf.nn.softmax(fc1)
        return prob


    def fc_layer(self, bottom, n_weight, name, stddev):
        #assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[-1]
        #initer = tf.truncated_normal_initializer(stddev=0.01)
        initer = tf.truncated_normal_initializer(stddev=stddev)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        ## rescale weight by 0.1
        #    W = tf.mul(.1, W)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.0, shape=[n_weight], dtype=tf.float32))
        ## rescale biase by 0.1
        #    b = tf.mul(.1, b)
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc


    def crossDomainContrastiveLoss(self):

        def contrastiveLoss(input_fea_1, input_fea_2, simLabel, margin, lossName):
            # contrastive loss construction
            distance_positive = tf.reduce_sum(tf.square(input_fea_1 - input_fea_2), axis=1)
            distance_negative = tf.maximum(0., margin-distance_positive)


            simLabel = tf.reshape(simLabel, [-1])

            distance_contrastive = tf.add(tf.multiply(simLabel, distance_positive), tf.multiply(1. - simLabel, distance_negative))

            loss = tf.reduce_mean(distance_contrastive, axis=0, name=lossName)
            loss_summary = tf.summary.scalar(lossName, loss)

            return loss, loss_summary, distance_contrastive, distance_positive



        # contrastive loss for sketch
        self.loss_sketch, self.loss_sketch_summary, _, _ = contrastiveLoss(self.sketch_1, self.sketch_2, self.simLabel_sketch, self.margin, 'loss_sketch')

        # contrastive loss for shape
        self.loss_shape, self.loss_shape_summary, _, _ = contrastiveLoss(self.shape_1, self.shape_2, self.simLabel_shape, self.margin, 'loss_shape')


        # contrastive loss for sketch-shape 1
        self.loss_cross_1, self.loss_cross_summary_1, self.distance_cross_1, self.distance_positive_1 = contrastiveLoss(self.sketch_1, self.shape_1, self.simLabel_cross_1, self.margin, 'loss_cross_1')

        # contrastive loss for sketch-shape 2
        self.loss_cross_2, self.loss_cross_summary_2, self.distance_cross_2, self.distance_positive_2 = contrastiveLoss(self.sketch_2, self.shape_2, self.simLabel_cross_2, self.margin, 'loss_cross_2')






    def build_network(self):
        with tf.variable_scope('sketch') as scope:
            self.sketch_out_1, self.sketch_debug_1 = self.sketchNetwork(self.input_sketch_fea_1)
            scope.reuse_variables()
            self.sketch_out_2, self.sketch_debug_2 = self.sketchNetwork(self.input_sketch_fea_2)

        with tf.variable_scope('shape') as scope:
            self.shape_out_1, self.shape_debug_1 = self.shapeNetwork(self.shape_reshape_1)
            scope.reuse_variables()
            self.shape_out_2, self.shape_debug_2 = self.shapeNetwork(self.shape_reshape_2)

    def build_model(self):
        # input sketch placeholder
        self.input_sketch_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_sketch_fea_1')
        self.input_sketch_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_1')
        self.input_sketch_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_sketch_fea_2')
        self.input_sketch_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_2')

        # input shape placeholder
        self.input_shape_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_views_shape, self.inputFeaSize], name='input_shape_fea_1')
        self.input_shape_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_1')
        self.input_shape_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_views_shape, self.inputFeaSize], name='input_shape_fea_2')
        self.input_shape_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_2')


        # reshape feature for forwarding through network
        self.shape_reshape_1 = tf.reshape(self.input_shape_fea_1, [-1, self.inputFeaSize], name='shape_reshape_1')
        self.shape_reshape_2 = tf.reshape(self.input_shape_fea_2, [-1, self.inputFeaSize], name='shape_reshape_2')


        self.simLabel_sketch = tf.cast(tf.equal(self.input_sketch_label_1, self.input_sketch_label_2), tf.float32, name='simLabel_sketch')
        self.simLabel_shape = tf.cast(tf.equal(self.input_shape_label_1, self.input_shape_label_2), tf.float32, name='simLabel_shape')


        self.simLabel_cross_1 = tf.cast(tf.equal(self.input_sketch_label_1, self.input_shape_label_1), tf.float32, name='simLabel_cross_1')
        self.simLabel_cross_2 = tf.cast(tf.equal(self.input_sketch_label_2, self.input_shape_label_2), tf.float32, name='simLabel_cross_2')



        self.build_network()


        if self.lossType == 'weightedFeaContrastiveLoss':
            print("build weightedFeaContrastiveLoss")

            self.sketch_1 = tf.reshape(self.sketch_out_1, [self.batch_size, self.outputFeaSize])
            self.sketch_2 = tf.reshape(self.sketch_out_2, [self.batch_size, self.outputFeaSize])


            # shape features before weighted summation
            self.shape_1_ = tf.reshape(self.shape_out_1, [self.batch_size, self.num_views_shape, self.outputFeaSize])
            self.shape_2_ = tf.reshape(self.shape_out_2, [self.batch_size, self.num_views_shape, self.outputFeaSize])


            self.shape_concatenateViews_1 = tf.reshape(self.shape_out_1, [self.batch_size, self.num_views_shape*self.outputFeaSize])
            self.shape_concatenateViews_2 = tf.reshape(self.shape_out_2, [self.batch_size, self.num_views_shape*self.outputFeaSize])


            with tf.variable_scope('classify') as scope:
                self.prob_1 = self.weightNet(self.shape_concatenateViews_1)
                self.prob_1 = tf.expand_dims(self.prob_1, axis=-1)
                scope.reuse_variables()
                self.prob_2 = self.weightNet(self.shape_concatenateViews_2)
                self.prob_2 = tf.expand_dims(self.prob_2, axis=-1)



            self.shape_1 = tf.reduce_sum(tf.multiply(self.prob_1, self.shape_1_), axis=1)
            self.shape_2 = tf.reduce_sum(tf.multiply(self.prob_2, self.shape_2_), axis=1)


            # self.shape_1 = tf.reduce_mean(self.shape_1_, axis=1)
            # self.shape_2 = tf.reduce_mean(self.shape_2_, axis=1)


            # build contrastive loss
            self.crossDomainContrastiveLoss()

            # summation of all the loss
            self.loss = tf.add_n([self.loss_sketch, self.loss_shape, self.loss_cross_1, self.loss_cross_2], name='loss')
            self.loss_summary = tf.summary.scalar('loss', self.loss)

    def ckpt_status(self):
        print("[*] Reading checkpoint ...")
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.model_checkpoint_path = ckpt.model_checkpoint_path
            return True
        else:
            return None

    def train(self):
        self.gm_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=100)
        start_time = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)
            sess.run(init)

            if self.ckpt_status():
                print("[*] Load SUCCESS")
                saver.restore(sess, self.model_checkpoint_path)
            else:
                print("[*] Load failed")

            for iter in range(self.maxiter):
                # This is old loading data


                sketch_fea_1, sketch_label_1 = self.nextBatch(self.batch_size, 'sketch_train')
                shape_fea_1, shape_label_1 = self.nextBatch(self.batch_size, 'shape')

                sketch_fea_2, sketch_label_2 = self.nextBatch(self.batch_size, 'sketch_train')
                shape_fea_2, shape_label_2 = self.nextBatch(self.batch_size, 'shape')

                if self.lossType == 'weightedFeaContrastiveLoss':
                    _, loss_, loss_sum_, sketch_fea, shape_fea, simLabel_sketch, simLabel_shape, simLabel_cross_1, simLabel_cross_2 = sess.run([self.gm_optim, self.loss, self.loss_summary, self.sketch_1, self.shape_1, self.simLabel_sketch, self.simLabel_shape, self.simLabel_cross_1, self.simLabel_cross_2], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                            })


                writer.add_summary(loss_sum_, iter)
                # reset u values
                if iter % 500  == 0:       # every 10 batches, update sinkhorn
                    print("Iter: [%5d] [# examples: %5d] time: %4.4f, loss: %.8f" % (iter, self.shape_num, time.time() - start_time, loss_))

                # This is for debuging, not saving the checkpoint
                if iter % 5000 == 0:
                    saver.save(sess, os.path.join(self.ckpt_dir, self.ckpt_name), global_step=iter)
                    self.evaluation_online(sess)
                # self.evaluation_online(sess)

    def evaluation(self):
        fid = open(self.mapFile, 'w')
        self.getLabel()

        # initialize all the array to evaluation
        testSketchNumber = len(self.sketch_test_label)
        trainShapeNumber = len(self.shape_label)

        sketchMatrix = np.zeros((testSketchNumber, self.outputFeaSize))
        shapeMatrix = np.zeros((trainShapeNumber, self.outputFeaSize))
        viewSelectMatrix = np.zeros(trainShapeNumber)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        start_time = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)
            sess.run(init)

            if self.ckpt_status():
                print("[*] Load SUCCESS")
                saver.restore(sess, self.weightFile)
            else:
                print("[*] Load failed")
                exit()


            start_time = time.time()

            num_of_batch = int(testSketchNumber / self.batch_size)
            rem = testSketchNumber % self.batch_size
            # For sketch
            for i in range(0, num_of_batch * self.batch_size, self.batch_size):
                tmp = sess.run(self.sketch_1, feed_dict={self.input_sketch_fea_1: self.sketchTestFeaset[i:i+self.batch_size]})
                sketchMatrix[i:i+self.batch_size] = tmp

            if rem:
                tmp = sess.run(self.sketch_1, feed_dict={self.input_sketch_fea_1: self.sketchTestFeaset[-self.batch_size:]})
                sketchMatrix[-rem:] = tmp[-rem:]



        # For shape

            num_of_batch = int(trainShapeNumber / self.batch_size)
            rem = trainShapeNumber % self.batch_size
            for i in range(0, num_of_batch * self.batch_size, self.batch_size):
                tmp, tmp_prob = sess.run([self.shape_1, self.prob_1], feed_dict={self.input_shape_fea_1: self.shapeFeaset[i:i+self.batch_size]})
                viewSelectMatrix[i:i+self.batch_size] = np.argmax(tmp_prob, axis=1).reshape(-1)
                shapeMatrix[i:i+self.batch_size] = tmp


            rem = trainShapeNumber % self.batch_size

            if rem:
                tmp, tmp_prob = sess.run([self.shape_1, self.prob_1], feed_dict={self.input_shape_fea_1: self.shapeFeaset[-self.batch_size:]})
                viewSelectMatrix[-rem:] = np.argmax(tmp_prob, axis=1).reshape(-1)[-rem:]
                shapeMatrix[-rem:] = tmp[-rem:]

            distM = distance.cdist(sketchMatrix, shapeMatrix)

            model_label = np.array(self.shape_label).astype(int)
            test_label = np.array(self.sketch_test_label).astype(int)
            C_depths = self.retrievalParamSP()
            C_depths = C_depths.astype(int)
            # nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, model_label, test_label, testMode=1)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation(C_depths, distM, model_label, test_label, testMode=1)

            print 'The NN is %5f' % (nn_av)
            print 'The FT is %5f' % (ft_av)
            print 'The ST is %5f' % (st_av)
            print 'The DCG is %5f' % (dcg_av)
            print 'The E is %5f' % (e_av)
            print 'The MAP is %5f' % (map_)

            fid.write('The NN is %5f\n' % (nn_av))
            fid.write('The FT is %5f\n' % (ft_av))
            fid.write('The ST is %5f\n' % (st_av))
            fid.write('The DCG is %5f\n' % (dcg_av))
            fid.write('The MAP is %5f\n' % (map_))

            pre = np.expand_dims(pre, axis=1)
            rec = np.expand_dims(rec, axis=1)
            rec_pre = np.concatenate([rec, pre], axis=1)
            np.savetxt('precision_recall.txt', rec_pre, fmt='%.5f')


            print("Generating the retrieval list")
            for i in range(self.sketch_test_num):
                print(os.path.join(self.returnDir, self.sketch_test_data[i].split(' ')[0].rstrip('\n')))
                splitFileName = self.sketch_test_data[i].split(' ')[0].rstrip('\n').split('/')
                filename = splitFileName[-3] + '_' + splitFileName[-2] + '_' + splitFileName[-1].split('.')[0] + '.txt'
                print(filename)
                print(os.path.join(self.returnDir, filename))
                fid_returnList = open(os.path.join(self.returnDir, filename), 'w')
                for j in range(rankArray.shape[1]):
                    if int(viewSelectMatrix[int(rankArray[i,j])]) < 10:
                        shape_name = self.shape_data[int(rankArray[i,j])][0].split(' ')[0][:-7] + '00' + str(int(viewSelectMatrix[int(rankArray[i,j])])) + '.jpg'
                    elif int(viewSelectMatrix[int(rankArray[i,j])]) >= 10:
                        shape_name = self.shape_data[int(rankArray[i,j])][0].split(' ')[0][:-7] + '0' + str(int(viewSelectMatrix[int(rankArray[i,j])])) + '.jpg'
                    print(shape_name)
                    fid_returnList.write(shape_name + '\n')
                    # print(self.shape_data[int(rankArray[i,j])][0].split(' ')[0] + '\n')
                    # fid_returnList.write(self.shape_data[int(rankArray[i,j])][0].split(' ')[0] + '\n')
                fid_returnList.close()

    def evaluation_online(self, sess):

        self.getLabel()

        # initialize all the array to evaluation
        testSketchNumber = len(self.sketch_test_label)
        trainShapeNumber = len(self.shape_label)

        sketchMatrix = np.zeros((testSketchNumber, self.outputFeaSize))
        shapeMatrix = np.zeros((trainShapeNumber, self.outputFeaSize))
        viewSelectMatrix = np.zeros(trainShapeNumber)

        start_time = time.time()

        num_of_batch = int(testSketchNumber / self.batch_size)
        rem = testSketchNumber % self.batch_size
        # For sketch
        for i in range(0, num_of_batch * self.batch_size, self.batch_size):
            tmp = sess.run(self.sketch_1, feed_dict={self.input_sketch_fea_1: self.sketchTestFeaset[i:i+self.batch_size]})
            sketchMatrix[i:i+self.batch_size] = tmp

        if rem:
            tmp = sess.run(self.sketch_1, feed_dict={self.input_sketch_fea_1: self.sketchTestFeaset[-self.batch_size:]})

            sketchMatrix[-rem:] = tmp[-rem:]



        # For shape

        num_of_batch = int(trainShapeNumber / self.batch_size)
        rem = trainShapeNumber % self.batch_size
        for i in range(0, num_of_batch * self.batch_size, self.batch_size):
            tmp = sess.run(self.shape_1, feed_dict={self.input_shape_fea_1: self.shapeFeaset[i:i+self.batch_size]})
            shapeMatrix[i:i+self.batch_size] = tmp


        rem = trainShapeNumber % self.batch_size

        if rem:
            tmp = sess.run(self.shape_1, feed_dict={self.input_shape_fea_1: self.shapeFeaset[-self.batch_size:]})
            shapeMatrix[-rem:] = tmp[-rem:]




        distM = distance.cdist(sketchMatrix, shapeMatrix)


        model_label = np.array(self.shape_label).astype(int)
        test_label = np.array(self.sketch_test_label).astype(int)
        C_depths = self.retrievalParamSP()
        C_depths = C_depths.astype(int)
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation(C_depths, distM, model_label, test_label, testMode=1)

        print 'The NN is %5f' % (nn_av)
        print 'The FT is %5f' % (ft_av)
        print 'The ST is %5f' % (st_av)
        print 'The DCG is %5f' % (dcg_av)
        print 'The E is %5f' % (e_av)
        print 'The MAP is %5f' % (map_)
