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
		return ("in:" + str(self.get_input()) + " out:" + self.get_output())
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
	def get_batch(self, num):
		batch = []
		rand = randint(0, training_set.size() - 1)
		for i in range(num):
			if (i + num) >= self.size():
				num = -i
			batch.append(get_example(i + rand))
		return batch
				
def gradient(hypothesis, training_set, feature=-1, batch=0):
	if batch == 0:
		batch = training_set.size()
	total = 0
	if batch == training_set.size():
		for i in range(training_set.size() - 1):
			grad = hypothesis(training_set.get_input(i)) - training_set.get_output(i)
			if feature == -1:
				total += grad
			else:
				total += (grad * training_set.get_input_feature(i, feature))
	else:
		for i in training_set.get_batch(batch):
			total += hypothesis(i.get_all_inputs()) - i.get_output() 
	return (total/batch)

def mult_sum(list1, list2):
	total = 0
	for l1, l2 in zip(list1, list2):
		total += l1 * l2
	return total

def generate_training_set(function, num_points, low=1, high=100, variance = .05):
	num_features = function.__code__.co_argcount	
	ts = Training_Set([])
	for i in range(num_points):
		x = []
		for f in range(num_features):
			x.append(random.uniform(low, high))
		y = function(*x) + random.uniform(-variance * num_points / num_features, variance * num_points / num_features);
		ts.add(Training_Example(x, y))
	return ts

def mean(numbers):
	return (float(sum(numbers)) / max(len(numbers), 1))

def plot_hypothesis(equation, low=1, high=100, scale=1, symbol="r-"):
	x = []
	y = []
	num_features = equation.__code__.co_argcount
	for i in range(high - low):
		x.append((i + low) / scale)
		y.append(equation([x[-1]]))
	plt.plot(x, y, symbol)
	
def plot_function(equation, low=1, high=100, scale=1, symbol="r-"):
	x = []
	y = []
	num_features = equation.__code__.co_argcount
	for i in range(high - low):
		x.append((i + low) / scale)
		y.append(equation(x[-1]))
	plt.plot(x, y, symbol)

def plot_list(list1, list2, symbol="b*"):
	plt.plot(list1, list2, symbol)
	
def show_plot():
	plt.draw()
	plt.show()
		
