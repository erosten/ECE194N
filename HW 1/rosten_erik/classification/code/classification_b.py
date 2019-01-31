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

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

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
    print ("Final Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
    is_correct = correct_prediction.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})

    false_indices = np.where(is_correct == False)[0]
    bad_images = mnist.test.images[:3000][false_indices]
    bad_image_labels = mnist.test.labels[:3000][false_indices]

    for j in range(0,false_indices.shape[0]):
        false_index = false_indices[j]
        img = bad_images[j].reshape(28,28)
        correct_label = np.nonzero(bad_image_labels[j,:])[0]
        predicted_label = my_prediction.eval({x: bad_images[j].reshape(1,784)})
        imgplot = plt.imshow(img, cmap='gray')
        plt.title('Correct Label: {}, Predicted Label: {}'.format(correct_label, predicted_label))
        plt.show()


