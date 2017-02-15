import common as ml
from random import randint
from math import exp

class Classification(ml.Learn):

	#softmax has to know the given feature and the probability to find, and each set of parameters is for a set outcome
	#features x outcomes
	#parameters is a list of list; each list in parameters is a set of outcomes for each feature
	def linear(self, y, x): 
		"""Linear hp(x), multiplies a list of parameters and a list of inputs together"""
		total = 0
		for p, x in zip(y, x):
			total += p * x
		return total

	def softmax(self, inputs, outcome=None):
		if (outcome != None and outcome < len(self.params)):
			explin = exp(1 * self.linear(self.params[outcome], inputs))
			return (explin / (explin + 1))
		else:
			ret = []
			for i in self.params:
				explin = exp(1 * self.linear(i, inputs))
				ret.append(explin / (explin + 1))
			return ret
	#TODO the whole examples thing isn't object oriented, there should be a get_random_batch method in TrainingSet that returns a smaller TrainingSet
	def update_rule(self, class_, feature, examples):
		"""Feature i class j update rule for examples in training set examples	
		return float to change by"""
		grad =  0
		for i in examples:
			#TODO add weight
			diff = (1 if (self.ts.get_output(i) == class_) else 0) - self.regression_function(self.ts.get_input(i), class_)
			grad += self.ts.get_input_feature(i, feature) * diff
		return (grad/max(len(examples), 1))

	def __init__(self, train):
		#logistic is technically a subclass of softmax
		super().__init__(train)
		self.regression_function = self.softmax
		self.step_size = .01
		self.params = []
		self.hypothesis = self.softmax
	def error(self):
		"""Return error of function"""
		return 0
	def general(self, num_steps=100, batch=50):
		self.ts.pad_ones()
		#first index is class, second one is feature
		#params[class_][feature]
		for i in range(int(self.ts.get_num_classes())):
			list_ = ([0] * int(self.ts.get_num_features()))
			self.params.append(list_)	

		self.hypothesis = self.softmax
		examples = []
		for b in range(batch):
			examples.append(randint(0, self.ts.size() - 1))
		for step in range(num_steps - 1):
			#don't have this line for stochastic
			modify_params = self.params[:]
			for class_ in range(int(self.ts.get_num_classes() - 1)):
				for feature in range(self.ts.get_num_features()):
					modify_params[class_][feature] += self.update_rule(class_, feature, examples)
			self.params = modify_params;
	@staticmethod
	def test():
		return 0	


				
