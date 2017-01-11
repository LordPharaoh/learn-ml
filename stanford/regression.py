import ml_common as ml
from random import randint
from math import exp

#error (not gradient)
def lms_error(training_set, hypothesis):
	total = 0;
	for i in range(training_set.size()):
		total += (hypothesis(training_set.get_input(i)) - training_set.get_output(i)) ** 2
	return total / (2 * training_set.size())

def locally_weighted(x, bandwidth, step_size=.01):
	"""Returns a weight function for a given x input and a bandwidth.
	
	Keyword Arguments:
	x -- list of numbers, location to find line fpr
	bandwidth -- tau, controls width of bell

	Return Keyword Arguments:
	xi -- list of numbers, input from training set
	feature -- feature number being tested
	"""
	weight = lambda xi, feature: exp((-(xi[feature] - x[feature])**2)/(2*bandwidth**2))
	return lambda training_set, hypothesis, i, examples: ml.gradient(training_set, hypothesis, i, examples, step_size, weight)

def constant_weight(*args):
	return 1

def batch_gradient_descent(step_size=.01):
	return lambda training_set, hypothesis, i, examples: ml.gradient(training_set, hypothesis, i, examples, step_size, constant_weight)
	
def linear(training_set, 
			num_steps=100, 
			batch=50,
			update_rule=batch_gradient_descent(),
			log_level=ml.NORMAL, 
			log_frequency=100):
	"""Returns a function that accepts a list of inputs and returns a predicted output in accord with given data

	Arguments:
	training_set -- training_set to train with
	num_steps -- num steps to take towards minimum error before returning solution
	batch -- size of batch to train with each time
	update_rule -- function derivative of cost with weight or step size
		has to take four arguments: the training set, the function hypothesis, the feature number, and a list of example numbers to train with
	log_level -- amount of stuff to log
	log_frequency -- will log every log_frequency steps
	>>> ts = ml.generate_training_set(lambda x: 4 * x, 10, -10, 10, 0)
	>>> function = regression.linear(ts)
	>>> function([2])
	7.999999999999996
	>>> function([7])
	27.999999999999986
	>>> 
	>>> ts = ml.generate_training_set(lambda x, y: 4 * x + y, 10, -10, 10, 0)
	>>> function = regression.linear(ts)
	>>> function([3, 4])
	15.999999999884412
	>>> 
	"""
	#TODO doesn't handly intercepts properly, padding ones doesn't work
	#training_set.pad_ones()
	params = [0] * training_set.get_num_features()

	hypothesis = lambda x: ml.mult_sum(params, x)
	for step in range(num_steps):

		#get a batch of random training examples to train with
		#TODO there are probably better ways of doing this
		examples = []
		for d in range(batch):
			examples.append(randint(0, training_set.size() - 1))

		for i in range(training_set.get_num_features()):
			#update gradient
			params[i] -= update_rule(training_set, hypothesis, i, examples)

		hypothesis = lambda x: ml.mult_sum(params, x)

		#log stuff
		if step % log_frequency == 0:
			ml.log("Error " + str(step) + ": " + str(lms_error(training_set, hypothesis)), ml.V_VERBOSE, log_level)
		if step % log_frequency == 0:
			ml.log("Hypothesis " + str(params), ml.V_VERBOSE, log_level)

	return hypothesis
