import tensorflow as tf
import numpy as np
import os
import sys 
tf.set_random_seed(222)
np.random.seed(222)

def toy(input, tau):
    with tf.variable_scope('toy') as scope:
        inputFeaSize =  input.get_shape()[-1]
        w1 = tf.get_variable('w1', shape=[inputFeaSize, 5], initializer=tf.truncated_normal_initializer(stddev=0.01))
        out1 = tf.maximum(tau, tf.matmul(input, w1))            # modified relu ----> relu(x) = x if x > tau otherwise tau

        w2 = tf.get_variable('w2', shape=[5, 3], initializer=tf.truncated_normal_initializer(stddev=0.01))
        out2 = tf.maximum(tau, tf.matmul(out1, w2))            # modified relu ----> relu(x) = x if x > tau otherwise tau

        w3 = tf.get_variable('w3', shape=[3, 4], initializer=tf.truncated_normal_initializer(stddev=0.01))
        out3 = tf.matmul(out2, w3)            # modified relu ----> relu(x) = x if x > tau otherwise tau

        return out3





def iter(input, lamb):
    with tf.variable_scope('iter') as scope:
        sqrt_feaSize = 2
        # Take batch size as 1

        u0 = tf.constant(1., shape=[sqrt_feaSize, 1])
        inputMatrix = tf.reshape(input, [sqrt_feaSize,  sqrt_feaSize])
        inputMatrix_tr = tf.transpose(inputMatrix, perm=[1, 0], name='transpose')
        u = tf.get_variable(name='u', shape=[sqrt_feaSize, 1], initializer=tf.constant_initializer(1))
        v = tf.get_variable(name='v', shape=[sqrt_feaSize, 1], initializer=tf.constant_initializer(1))
        op_assign_v = v.assign(tf.div(u0, tf.matmul(inputMatrix_tr, u, name='mul_v')))
        op_assign_u = u.assign(tf.div(u0, tf.matmul(inputMatrix, v, name='mul_u')))
        #op_assign_u = u.assign(u+2)
        T = tf.matmul(tf.matmul(tf.diag(tf.reshape(u, [-1])), tf.exp(tf.multiply(-lamb, inputMatrix))), tf.diag(tf.reshape(v, [-1])))
        return op_assign_v, op_assign_u, u, v, T


    
def opt(input, T):
    with tf.variable_scope('opt') as scope:
        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(input, tf.reshape(T, [-1])), axis=1))
        return loss



t1 = tf.Variable(tf.truncated_normal(shape=[5, 2, 3]))
t2 = tf.Variable(tf.truncated_normal(shape=[5, 3, 2]))
t12 = tf.matmul(t1, t2)

kkkk = tf.get_variable("kk", shape=[3,1], initializer=tf.constant_initializer(1))

x = tf.placeholder(tf.float32, shape=[3, 6], name='x')
tau = 0.00005
lamb = 0.01

M = toy(x, tau)
op_assign_v, op_assign_u, u, v, T = iter(M[0], lamb)

loss = opt(M, T)

var_list = tf.trainable_variables()
for var in var_list:
    print(var.name)
grad_var = [ var for var in var_list if 'toy' in var.name]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, var_list=grad_var)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
ckpt_dir = './checkpoint'
ckpt_name = 'model'
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("./logs", sess.graph)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    t1_ = t1.eval()
    t2_ = t2.eval()
    t12_ =  t12.eval()
    print(t1_)
    print(t2_)
    print(t12_)
    print("VERIFY")

    for i in range(5): 
        print(np.equal(np.matmul(t1_[i], t2_[i]), t12_[i]))
    sys.exit()


    """
    if ckpt and ckpt.model_checkpoint_path:
        print("Load checkpoint [!!!]")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Load Failed [!!!]")
    """
    print("Start training")

    for i in range(10000):
        print(u.eval())
        print(v.eval())
        data_x = np.random.random((3,6))
        M_eval = M.eval(feed_dict={x: data_x})
        print("M_eval: {}".format(M_eval))
        for j in range(1000):
            op_assign_v.eval(feed_dict={M:M_eval})
            op_assign_u.eval(feed_dict={M:M_eval})
            if j % 100 == 0:
                print(u.eval())
                print(v.eval())
        _, loss_, T_, M_ = sess.run([optimizer, loss, T, M], feed_dict={x: data_x})
        print("M_: {}".format(M_))
        print("loss_: {:5.8f}".format(loss_))
        print("T_: {}".format(T_))







        #if not (i % 10):
        #    saver.save(sess, os.path.join(ckpt_dir, ckpt_name))

