#!/usr/bin/env python
# -*- coding:utf-8 -*
import tensorflow as tf

state = tf.Variable(0,name='counter')
print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for _ in range(3):
        session.run(update)
        print(session.run(state))
