from random import randint
import random
import gradient
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
	def __init__(self, function, num_points=30, low=-10, high=10, variance = .05):
		"""If they give a list of training examples, just add that first"""
		if (isinstance(function, list)):
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
		self.equation = function
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
	def plot(self, num=0, scale=1, symbol="b*"):
		plt.plot(self.get_plottable_input(num), self.get_plottable_output(), symbol)
		x = []
		y = []
		num_features = self.equation.__code__.co_argcount
		for i in range(round((self.domain(num)[1] - self.domain(num)[0])/scale)):
			x.append(i * scale + self.domain(num)[0])
			y.append(self.equation(x[-1]))
		plt.plot(x, y, "b-")
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
	def get_num_classes(self):
		classes = []
		num_classes = 0
		for i in self.examples:
			#if its not an int and greater than 0 (meaning class) then stop
			if (not float(i.get_output()).is_integer()) or i.get_output() < 0:
				raise TypeError("Classification requires whole-number classes that are greater than 0.")
				#is it really necessary to return after raising an error? Who knows
				return
		#add one for 0
		return max(self.get_plottable_output()) + 1

class Learn:
	"""Abstract class for learning algorithms
	"""	
	def __init__(self, train):
		"""train: training set to regress"""
		self.ts = train
		self.parameters = []
		self.hypothesis = lambda x: x	

	"""Requirements for weight functions
	A function which takes two arguments: an x value and a feature number
	The function should return a float weight value

	A weight function generator should return such a function
	"""
	@staticmethod
	def generate_local_weight(self, x, bandwidth=1):
		#TODO locally weighted regression is totally broken
		"""Returns a weight function for a given x input and a bandwidth.
		
		Keyword Arguments:
		x -- location to find line fpr
		bandwidth -- tau, controls width of bell

		Return Keyword Arguments:
		xi -- list of numbers, input from training set
		feature -- feature number being tested
		"""
		weight = lambda xi, feature: exp((-(xi[feature] - x)**2)/(2*bandwidth**2))
		return weight

	@staticmethod
	def constant_weight(*args):
		return 1
	def set_local_weight(self, step_size, x, bandwidth=1):
		self.update_rule = gradient.batch_descent(step_size, Regression.generate_local_weight(x, bandwidth))
	
	def evaluate(x):
		return self.hypothesis(x)
	def test():
		raise NotImplementedError("not implemented in child class")
		return 0
	def get_params(self):
		return self.params
	def plot(self, i, scale=1):
		""" Plots ith feature with training set """	
		self.ts.plot(i)
		bounds = self.ts.domain(i)
		x = [] 
		y = []
		num_features = self.hypothesis.__code__.co_argcount
		times = round(((bounds[1] - bounds[0]) / scale))
		for j in range(times):
			x.append(j * scale + bounds[0])
			vec = [0] * (num_features + 1)
			vec[-1] = 1
			vec[i] = x[-1]
			y.append(self.hypothesis(vec))
		plt.plot(x, y, "r-")
		


		
		
				
def mean(numbers):
	return (float(sum(numbers)) / max(len(numbers), 1))

def show_plot():
	plt.draw()
	plt.show()		
