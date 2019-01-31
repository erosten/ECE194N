import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# iterate through digits
# odd digit 1's are in index 1
# even digit 1's are in index 0
def map_labels(labels):
    new_labels = np.zeros((labels.shape[0],2))
    for i in range(0,10):
        indices = np.where(labels[:,i] == 1)
        if (i % 2 == 0):
            new_labels[indices,0] = 1
        else:
            new_labels[indices,1] = 1
    return new_labels 


images = mnist.test.images[:3000]
labels = map_labels(mnist.test.labels[:3000])


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784

y = tf.placeholder(tf.float32, [None, 2]) # even / odd recognition ==> 2 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_ys = map_labels(batch_ys)
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        my_prediction = tf.argmax(pred,1)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


    print("Optimization Finished!")
    print ("Final Accuracy:", accuracy.eval({x: images, y: labels}))



