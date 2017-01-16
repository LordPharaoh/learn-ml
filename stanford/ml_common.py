from random import randint
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLORS = ("b", "g", "c", "y", "m", "k") 

QUIET = 0
NORMAL = 1
VERBOSE = 2
V_VERBOSE = 3

#global log state, can also have function-specific ones
LOG_STATE = NORMAL

def log(s, level, state=LOG_STATE):
	if state >= level:
		print(str(s))

class Training_Example:
	#should be a list of features and an output
	def __init__(self, x, y):
		self.x = x;
		self.y = y;
	def get_all_inputs(self):
		return self.x
	def get_input(self, num):
		return self.x[num]
	def get_output(self):
		return self.y
	def size(self):
		return len(self.x)
	def pad_one(self):
		self.x.insert(0, 1)
	def __str__(self):
		return (str(self.get_all_inputs()) + " " + str(self.get_output()))
class Training_Set:
	#should be a list of training examples
	def __init__(self, x):
		self.examples = x;
	#padding ones makes the y-intercept easier later
	def pad_ones(self):
		for i in self.examples:
			i.pad_one()
	def size(self):
		return len(self.examples)
	#get one feature of x so you can plot it in matplotlib
	def get_plottable_input(self, feature):
		f = []
		for i in self.examples:
			f.append(i.get_input(feature))
		return f
	#get list of y to plot in matplotlib
	def get_plottable_output(self):
		f = []
		for i in self.examples:
			f.append(i.get_output())
		return f
	def add(self, e):
		self.examples.append(e)
	def get_example(self, num):
		return self.examples[num]
	def get_input(self, num):
		return self.examples[num].get_all_inputs()
	def get_input_feature(self, num, num1):
		return self.examples[num].get_input(num1)
	def get_num_features(self):
		return self.get_example(0).size()
	def get_output(self, num):
		return self.examples[num].get_output()
	#plot one dimension
	def plot(self, num, symbol="b*"):
		plt.plot(self.get_plottable_input(num), self.get_plottable_output(), symbol)
	def __str__(self):
		stb = ""
		for i in self.examples:
			stb += str(i) + ";"
		return stb
	def get_batch(self, num):
		batch = []
		rand = randint(0, training_set.size() - 1)
		for i in range(num):
			if (i + num) >= self.size():
				num = -i
			batch.append(get_example(i + rand))
		return batch
				
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
	>>> ts = ml.generate_training_set(lambda x: 4*x, 10, -10, 10, variance=0)
	>>> hypothesis = lambda x: 3 * x[0]
	>>> ml.gradient(ts, hypothesis, 0, range(ts.size() - 1), .1)
	-3.4423139184132863
	>>> hypothesis = lambda x: 4 * x[0]
	>>> ml.gradient(ts, hypothesis, 0, range(ts.size() - 1), .1)
	0.0
	>>> 
	"""

	total = 0
	for i in examples:
		grad = hypothesis(training_set.get_input(i)) - training_set.get_output(i)
		total += (grad * training_set.get_input_feature(i, feature) * weight_function(training_set.get_input(i), feature) * step_size)
	#don't divide by zero, the answer will be 0 anyway, just divide by one
	return (total/max(len(examples), 1))

def generate_training_set(function, num_points, low=1, high=100, variance = .05):
	"""Return a training set of a specific function

	Arguments:
	function -- function to generate, can take any number of args and returns a float
	num_points -- number of points to generate
	low -- low bound
	high -- high bound
	variance -- randomness in the data

	NOT SUITABLE FOR DOCTEST, CONTAINS RANDOM DATA
	!>>> ts = ml.generate_training_set(lambda x: 4*x, 3, -10, 10, variance=0) 
	!>>> str(ts)
	'[5.664584629276554] 22.658338517106216;[-0.16071175241091495] -0.6428470096436598;[-8.627610821082763] -34.51044328433105;'
	"""
	num_features = function.__code__.co_argcount	
	ts = Training_Set([])
	for i in range(num_points):
		x = []
		for f in range(num_features):
			x.append(random.uniform(low, high))
		y = function(*x) + random.uniform(-variance * (num_points / num_features), variance * (num_points / num_features));
		ts.add(Training_Example(x, y))
	return ts

def mean(numbers):
	return (float(sum(numbers)) / max(len(numbers), 1))

#TODO plot_hypothesis and plot_function are super similar and should be one thing
def plot_hypothesis(equation, low=1, high=100, scale=1, symbol="r-"):
	x = []
	y = []
	num_features = equation.__code__.co_argcount
	times = round(((high - low) / scale))
	for i in range(times):
		x.append(i * scale + low)
		y.append(equation([x[-1]]))
	plt.plot(x, y, symbol)
	
def plot_function(equation, low=1, high=100, scale=1, symbol="r-"):
	x = []
	y = []
	num_features = equation.__code__.co_argcount
	for i in range(round((high - low)/scale)):
		x.append(i * scale + low)
		y.append(equation(x[-1]))
	plt.plot(x, y, symbol)

def plot_list(list1, list2, symbol="b*"):
	plt.plot(list1, list2, symbol)
	
def show_plot():
	plt.draw()
	plt.show()
		
