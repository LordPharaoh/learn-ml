import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])
#We want to have multiple possible weights and biases
#the neurons (yes/no spitters) need a slightly positive starting value so we will init them in this function
def weight(shape):
	#truncated normal curve with a standard deviation of 1
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias(shape):
	#constant bias
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#convolutions, part of the image we pass through the net first
#keep default strides
#0 padded. output same as input
def conv2d(x1, W1):
	return tf.nn.conv2d(x1, W1, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')


#first layer
#32 features for each 5x5 bit, overlapp
W_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])

#image
#1 & 2 are w and h and 3 is the number of color channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

#convolve image with weight and add bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#attach next neural network to the first one

#first layer
#64 features for each 5x5 bit, this time it's a 5x1 image because we're hooking up the result of the earlier net
W_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])

#now actually do the relu
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#now connect everything together in one image that we can process
#coz now it's a 7x7 image because each layer of convnet split the size in half
W_fc1 = weight([7*7*64, 1024])
b_fc1 = bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout and thin network by randomly taking out things while training to prevent overfitting
#chance we keep something and don't get rid of it
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight([1024, 10])
b_fc2 = bias([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

sess = tf.Session()
#multiply each element with predicted answer, get the sum, then calculate the mean
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#train with ADAM, better than gradient descent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#run and train
result = sess.run(tf.global_variables_initializer())
sess.as_default()
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)

print("test accuracy %g"%accuracy.eval(feed_dict={
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

