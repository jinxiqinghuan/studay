# import tensorflow as tf
# import timeit
#
# with tf.device('/cpu:0'):
#     cpu_a = tf.random.normal([10000, 1000])
#     cpu_b = tf.random.normal([1000, 2000])
#     print(cpu_a.device, cpu_b.device)
#
# with tf.device('/gpu:0'):
#     gpu_a = tf.random.normal([10000, 1000])
#     gpu_b = tf.random.normal([1000, 2000])
#     print(gpu_a.device, gpu_b.device)
#
# def cpu_run():
#     with tf.device('/cpu:0'):
#         c = tf.matmul(cpu_a, cpu_b)
#     return c
#
# def gpu_run():
#     with tf.device('/gpu:0'):
#         c = tf.matmul(gpu_a, gpu_b)
#     return c
#
# cpu_time = timeit.timeit(cpu_run(), number=10)
# gpu_time = timeit.timeit(gpu_run(), number=10)
#
# # import numpy as np
# # import tensorflow as tf
# #
# # print(np.zeros(5))
# # print("hello")
# #
#
#







import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
a=tf.constant(2)
b=tf.constant(3)
with tf.Session() as sess:
    print("a:%i" % sess.run(a),"b:%i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constant:%i" % sess.run(a*b))





