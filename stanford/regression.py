import ml_common as ml
from random import randint
from math import exp

#error (not gradient)
def lms_error(training_set, hypothesis):
	total = 0;
	for i in range(training_set.size()):
		total += (hypothesis(training_set.get_input(i)) - training_set.get_output(i)) ** 2
	return total / (2 * training_set.size())

#TODO gradient can be optimized by taking it once, multiplying by each feature, and then returning the whole thing as a list
#contd. Getting the mean of the whole thing seemed to work, but it may give bad results with large training sets and small batch sizes
#so basically have an option to return a list of gradients for all features, this will be faster too
def gradient(training_set, hypothesis, feature, examples, step_size, weight_function=lambda xi, feature: 1):
    """Return the gradient of a number of examples in the training set with a given hypothesis function, step size, and weight.
    
    Keyword Arguments:
    training_set -- the training set
    hypothesis -- the prediction function. Should take in a list of inputs and return a prediction.
    feature -- the feature number 
    examples -- a list of examples in the training set to try it on; e.g. [1, 3, 7] will try it on the 1st, 3rd, and 7th training examples
    step_size -- constant to reduce the gradient by
    weight_function -- weight test points, change for locally weighted regression and other stuff
    Gradient 
        The sum of the difference of the predicted value (hypothesis) and the actual output (y) muliplied by the value of the input feature.                                    
        j = iteration value of the sum
        i = feature
        sum((h(x[j]) - y[j]) * x[j][i]) * weight * step_size
	>>> import regression
    >>> ts = ml.TrainingSet.generate(lambda x: 4*x, 10, -10, 10, variance=0)
    >>> hypothesis = lambda x: 3 * x[0]
    >>> regression.gradient(ts, hypothesis, 0, range(ts.size() - 1), .1)
    -3.4423139184132863
    >>> hypothesis = lambda x: 4 * x[0]
    >>> regression.gradient(ts, hypothesis, 0, range(ts.size() - 1), .1)
    0.0
    >>> 
    """
    
    total = 0
    for i in examples:
        grad = hypothesis(training_set.get_input(i)) - training_set.get_output(i)
        total += (grad * training_set.get_input_feature(i, feature) * weight_function(training_set.get_input(i), feature) * step_size)
    #don't divide by zero, the answer will be 0 anyway, just divide by one
    return (total/max(len(examples), 1))
    



"""Requirements for weight functions
A function which takes two arguments: an x value and a feature number
The function should return a float weight value

A weight function generator should return such a function
"""
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
	return weight

def constant_weight(*args):
	return 1

"""Requirements for descent functions
Take four arguments: training_set, hypothesis, feature number, and a numerical list of examples.
Return a float change value.

A descent function generator should return such a function.
"""

def batch_gradient_descent(step_size=.01, weight=constant_weight,):
	"""Returns batch gradient descent function with given step size
	The gradient function accepts a training set, hypothesis function, a feature, and a list of number examples
	"""
	return lambda training_set, hypothesis, i, examples: -gradient(training_set, hypothesis, i, examples, step_size, weight)

def batch_gradient_ascent(step_size=.01, weight=constant_weight,):
	"""Returns batch gradient descent function with given step size
	The gradient function accepts a training set, hypothesis function, a feature, and a list of number examples
	"""
	return lambda training_set, hypothesis, i, examples: gradient(training_set, hypothesis, i, examples, step_size, weight)

"""Requirements for regression type functions:
Take an argument of two lists, the first list a set of parameters and the second list a set of input variables.
Use these two lists to perform some function and return a single float.
"""

def linear(params, x):
	"""Linear hp(x), multiplies a list of parameters and a list of inputs together"""
	total = 0 
	for p, x in zip(params, x):
		total += p * x
	return total

def logistic(list1, list2):
	"""Logistic hp(x), do the linear thing and then exp that. (exp(lin)) / (exp(lin)+1)"""
	explin = exp(-1 * linear(list1, list2))
	return (explin / (explin + 1))

#TODO batch size should be an argument of the gradient descent function, not the regression itself
def general(training_set, 
			num_steps=100, 
			batch=50,
			update_rule=batch_gradient_descent(),
			regression_function=linear,
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
