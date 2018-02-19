# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 07:00:15 2017

Learn tf : Variables

@author: Atul
"""

#import
import tensorflow  as tf
import numpy as np

# native python
x = 35
y = x + 5
print(y)

# in tensorflow
x = tf.constant(35, name='x')
z = tf.constant([35, 40, 45], name='z')
y = tf.Variable(x+5, name='y')
a = tf.Variable(z+5, name='a')
print(y)


model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
    print(session.run(a))


X = np.random.randint(1000, size=1000)
s = tf.Session()
s.run(model)
for i in range(5):
    x = x+1
    print(s.run(x))

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("D:/Git/DeepLearning/learning_tensorflow/basic", session.graph)
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))
