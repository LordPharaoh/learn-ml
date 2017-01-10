import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#784 one-hot data possibilities
#x is a placeholder value
#MNIST images are inputted here
x = tf.placeholder(tf.float32, [None, 784])

#Tensors are like number-holders for many dimensions...
#at least that's how I vaguely understand them
#these tensorflow Variables are modifiable tensors in the tensorflow graph
#the model parameters are usually Variables

#ONE_HOT: vector of 1 in one dimension.
#10 one-hots in 784 dimensions???
#these are going to be learned eventually
#Weight
W = tf.Variable(tf.zeros([784, 10]))
#10 one-hots
#bias, 
b = tf.Variable(tf.zeros([10]))

#evidence: (Wx + b)
#one-line softmax
y = tf.matmul(x, W) + b

#define cost or loss of model
#-sum(y'log(y))
#y' : true distribution (vector labels)
#y : predicted distribution
y_ = tf.placeholder(tf.float32, [None, 10])
#mutiply each element of y_ with the corresponding predicted log
#get the sum of that then calculate the overall mean
#more numerically stable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#train with a chosen algorithm to minimize loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize created variables
init = tf.global_variables_initializer()

#run in session
sess = tf.Session()
sess.run(init)

#train 1000 times
#each time, the optimisation algorithm moves a little bit closer to the correct solution.
for i in range(1000):
	#get a batch of 100 random training points every time you loop
	#small batches of random data: stochastic training
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#testing for correctness
#tf.argmax(y, 1) is the predicted answer and tf.argmax(y_, 1) is the correct answer
#list of booleans of right answers
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#cast to floats and take the average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


