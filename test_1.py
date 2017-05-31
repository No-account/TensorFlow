#这段很短的 Python 程序生成了一些三维数据, 然后用一个平面拟合它.

import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
#np.random.rand(x,y)随机生成一个x*y且所有元素都在0-1之间的矩阵
x_data=np.float32(np.random.rand(2,100))
#np.dot()矩阵相乘
y_data=np.dot([0.100,0.200],x_data)+0.300


# 构造一个线性模型
#tf.zeros(shape, dtype=tf.float32, name=None)生成shape形状的tensor,所有元素为0,类似的还有tf.ones(),tf.ones_like(),tf.zeros_like()
b=tf.Variable(tf.zeros([1]))
#tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)均匀分布随机数，范围为[minval,maxval]
#类似的还有tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)正太分布随机数，均值mean,标准差stddev
#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
#matmul(a, b,transpose_a=False, transpose_b=False,a_is_sparse=False, b_is_sparse=False,name=None)tensor乘法可以看到还提供了transpose和is_sparse的选项。如果对应的transpose项为True，例如transpose_a=True,那么a在参与运算之前就会先转置一下。而如果a_is_sparse=True,那么a会被当做稀疏矩阵来参与运算
y=tf.matmul(w,x_data)+b


# 最小化方差
#tf.square(x, name=None)计算平方 (y = x * x = x^2)
#tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)对矩阵进行降维运算 ，在reduction_indices维度上取平均值,如果不指定第二个参数就是对所有的求平均值
#类似的还有tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
loss=tf.reduce_mean(tf.square(y-y_data))
#tf.train.GradientDescentOptimizer这个类是实现梯度下降算法的优化器__init__(learning_rate, use_locking=False,name=’GradientDescent’)learning_rate: A Tensor or a floating point value. 要使用的学习率 use_locking: 要是True的话，就对于更新操作（update operations.）使用锁 name: 名字，可选，默认是”GradientDescent”.
#minimize(loss,global_step=None,var_list=None,gate_gradients=GATE_OP,aggregation_method=None,colocate_gradients_with_ops=False,name=None)通过更新var_list来减小loss
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)


# 初始化变量
init=tf.initialize_all_variables()
#启动图
sess=tf.Session()
sess.run(init)


#运行这个计算图谱并且每迭代就打印出w和b
for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))
