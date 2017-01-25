import common as ml
import gradient 
from random import randint
from math import exp

#error (not gradient)
def lms_error(training_set, hypothesis):
	total = 0;
	for i in range(training_set.size()):
		total += (hypothesis(training_set.get_input(i)) - training_set.get_output(i)) ** 2
	return total / (2 * training_set.size())

"""Requirements for weight functions
A function which takes two arguments: an x value and a feature number
The function should return a float weight value

A weight function generator should return such a function
"""
#TODO locally weighted regression is totally broken
def locally_weighted(x, bandwidth=1):
	"""Returns a weight function for a given x input and a bandwidth.
	
	Keyword Arguments:
	x -- list of numbers, location to find line fpr
	bandwidth -- tau, controls width of bell

	Return Keyword Arguments:
	xi -- list of numbers, input from training set
	feature -- feature number being tested
	"""
	weight = lambda xi, feature: exp((-(xi[feature] - x)**2)/(2*bandwidth**2))
	return weight

def constant_weight(*args):
	return 1

"""Requirements for regression type functions:
Take an argument of two lists, the first list a set of parameters and the second list a set of input variables.
Use these two lists to perform some function and return a single float.
"""

def linear_function(params, x):
	"""Linear hp(x), multiplies a list of parameters and a list of inputs together"""
	total = 0 
	for p, x in zip(params, x):
		total += p * x
	return total


#TODO batch size should be an argument of the gradient descent function, not the regression itself
def general(training_set, 
			num_steps=100, 
			batch=50,
			update_rule=gradient.batch_descent(),
			regression_function=linear_function,
			log_level=ml.NORMAL, 
			log_frequency=100):
	"""Returns a function that accepts a list of inputs and returns a predicted output in accord with given data

	Arguments:
	training_set -- training_set to train with
	num_steps -- num steps to take towards minimum error before returning solution
	batch -- size of batch to train with each time
	update_rule -- function derivative of cost with weight or step size
		has to take four arguments: the training set, the function hypothesis, the feature number, and a list of example numbers to train with
	regression_function -- function to regress: e.g. linear, logistic. Function accepts a list of params and a list of inputs and returns a float output
	log_level -- amount of stuff to log
	log_frequency -- will log every log_frequency steps
	"""
	training_set.pad_ones()
	params = [0] * training_set.get_num_features()

	hypothesis = lambda x: regression_function(params, x)
	for step in range(num_steps - 1):

		#get a batch of random training examples to train with
		#TODO there are probably better ways of doing this
		examples = []
		for d in range(batch):
			examples.append(randint(0, training_set.size() - 1))
		for i in range(training_set.get_num_features()):
			#update gradient
			#always addition, update rule should return the correct sign
			params[i] += update_rule(training_set, hypothesis, i, examples)

		hypothesis = lambda x: regression_function(params, x)

		#log stuff
		if step % log_frequency == 0:
			ml.log("Error " + str(step) + ": " + str(lms_error(training_set, hypothesis)), ml.V_VERBOSE, log_level)
		if step % log_frequency == 0:
			ml.log("Hypothesis " + str(params), ml.V_VERBOSE, log_level)

	return hypothesis

def lwr(training_set, x, bandwidth=1, num_steps=100, step_size=.01, batch=50,log_level=ml.NORMAL, log_frequency=100):
	return general (training_set, num_steps, batch, gradient.batch_descent(step_size, locally_weighted(x, bandwidth)), linear_function, log_level, log_frequency)

def linear(training_set, num_steps=100, step_size=.01, batch=50,log_level=ml.NORMAL, log_frequency=100):
	return general(training_set, num_steps, batch, gradient.batch_descent(step_size, constant_weight), linear_function, log_level, log_frequency)
