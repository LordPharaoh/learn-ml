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

class TrainingExample:
	#should be a list of features and an output
	x = []
	y = []

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
		self.x.append(1)
	def __str__(self):
		return (str(self.get_all_inputs()) + " " + str(self.get_output()))

class TrainingSet:
	#generate one automatically
	def __init__(self, function, num_points=None, low=1, high=100, variance = .05):
		"""If they give a list of training examples, just add that first"""
		if (isinstance(function, list) and num_points == None):
			self.examples = function
			return
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
		self.examples = []
		num_features = function.__code__.co_argcount	
		for i in range(num_points):
			x = []
			for f in range(num_features):
				x.append(random.uniform(low, high))
			y = function(*x) + random.uniform(-variance * (num_points / num_features), variance * (num_points / num_features));
			self.add(TrainingExample(x, y))
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
	def plot(self, num=0, symbol="b*"):
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
	def domain(self, i):
		""" Returns domain, or a tuple with the min value of x at 0 and max at 1 """
		smallest = self.examples[0].get_input(i)
		biggest = self.examples[0].get_input(i)
		for e in self.examples:
			if e.get_input(i) > biggest: biggest = e.get_input(i)
			if e.get_input(i) < smallest: smallest = e.get_input(i)
		return (smallest, biggest)
	def range(self):
		""" Returns range, or a tuple with the min value of y at 0 and max at 1 """
		smallest = self.examples.get_output(i)
		biggest = self.examples.get_output(i)
		for e in self.examples:
			if e.get_output(i) > biggest: biggest = e.get_output(i)
			if e.get_output(i) < smallest: smallest = e.get_output(i)
		return (smallest, biggest)
		
		
				
def mean(numbers):
	return (float(sum(numbers)) / max(len(numbers), 1))

#TODO plot_hypothesis and plot_function are super similar and should be one thing
#TODO the whole hypothesis-function thing is a mess, they should be able to handle list, training set, and comma-seperated arguments intelligently
#TODO should handle choosing which feature to graph and fill in values for the other features
#just have like one plot function and let it intelligently deal with all scenarios
def plot_hypothesis(hyp, low=1, high=100, scale=1, symbol="r-"): 
	x = []
	y = []
	num_features = hyp.__code__.co_argcount
	times = round(((high - low) / scale))
	for i in range(times):
		x.append(i * scale + low)
		y.append(hyp([x[-1], 1]))
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
		
