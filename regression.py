import common as ml
import gradient 
from random import randint
from math import exp

class Regression(common.Learn):

	"""Requirements for regression type functions:
	Take an argument of two lists, the first list a set of parameters and the second list a set of input variables.
	Use these two lists to perform some function and return a single float.
	"""

	def linear_function(self, x):
		"""Linear hp(x), multiplies a list of parameters and a list of inputs together"""
		total = 0 
		for p, x in zip(self.params, x):
			total += p * x
		return total

	def __init__(self, train):
		super(Regression, self).init(train)
		self.regression_function = self.linear_function

	def lms_error(self):
		total = 0;
		for i in range(self.ts.size()):
			total += (self.hypothesis(self.ts.get_input(i)) - self.ts.get_output(i)) ** 2
		return total / (2 * self.ts.size())
		
	#TODO batch size should be an argument of the gradient descent function, not the regression itself
	def general(self,
				num_steps=100, 
				batch=50,
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
		self.ts.pad_ones()
		self.params = [0] * self.ts.get_num_features()

		self.hypothesis = lambda x: self.regression_function(x)
		for step in range(num_steps - 1):

			#get a batch of random training examples to train with
			#TODO there are probably better ways of doing this
			examples = []
			for d in range(batch):
				examples.append(randint(0, self.ts.size() - 1))
			for i in range(self.ts.get_num_features()):
				#update gradient
				#always addition, update rule should return the correct sign
				self.params[i] += self.update_rule(self.ts, self.hypothesis, i, examples)

			self.hypothesis = lambda x: self.regression_function(x)

			#log stuff
			if step % log_frequency == 0:
				ml.log("Error " + str(step) + ": " + str(self.lms_error()), ml.V_VERBOSE, log_level)
			if step % log_frequency == 0:
				ml.log("Hypothesis " + str(self.params), ml.V_VERBOSE, log_level)

	
	

