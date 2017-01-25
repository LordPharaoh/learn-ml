from random import randint
from math import exp
import common as ml
#TODO gradient can be optimized by taking it once, multiplying by each feature, and then returning the whole thing as a list
#contd. Getting the mean of the whole thing seemed to work, but it may give bad results with large training sets and small batch sizes
#so basically have an option to return a list of gradients for all features, this will be faster too
def gradient(training_set, hypothesis, feature, examples, step_size, weight_function=lambda xi, feature: 1):
    """Return the gradient of a number of examples in the training set with a given hypothesis function, step size, and weight.
    
    Keyword Arguments:
    training_set -- the training set hypothesis -- the prediction function. Should take in a list of inputs and return a prediction.
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
    
"""Requirements for descent functions
Take four arguments: training_set, hypothesis, feature number, and a numerical list of examples.
Return a float change value.

A descent function generator should return such a function.
"""
def batch_descent(step_size=.01, weight=lambda x: 1):
	"""Returns batch gradient descent function with given step size
	The gradient function accepts a training set, hypothesis function, a feature, and a list of number examples
	"""
	return lambda training_set, hypothesis, i, examples: -gradient(training_set, hypothesis, i, examples, step_size, weight)

def batch_ascent(step_size=.01, weight=lambda x:1,):
	"""Returns batch gradient descent function with given step size
	The gradient function accepts a training set, hypothesis function, a feature, and a list of number examples
	"""
	return lambda training_set, hypothesis, i, examples: gradient(training_set, hypothesis, i, examples, step_size, weight)


