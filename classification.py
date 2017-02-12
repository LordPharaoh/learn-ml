import common as ml

class Classification(ml.Learn):
	#TODO remove, logistic is a subset of softmax
	def logistic(list1, list2):
		"""Logistic hp(x), do the linear thing and then exp that. (exp(lin)) / (exp(lin)+1)"""
		explin = exp(1 * r.linear(list1, list2))
		return (explin / (explin + 1))

	#softmax has to know the given feature and the probability to find, and each set of parameters is for a set outcome
	#features x outcomes
	#parameters is a list of list; each list in parameters is a set of outcomes for each feature
	def softmax(parameters, inputs, outcome=None):
		if (outcome != None and outcome < len(parameters)):
			explin = exp(1 * r.linear(parameters[outcome], inputs))
			return (explin / (explin + 1))
		else:
			for i in parameters:
				explin = exp(1 * r.linear(i, inputs))
				return (explin / (explin + 1))
	def update_rule(i, j, examples):
		"""Feature i class j update rule for examples in training set examples
		return float to change by"""
		return 0

	def __init__(self, train):
		#logistic is technically a subclass of softmax
		super().__init__(train)
		self.regression_function = self.softmax
	def error(self):
		"""Return error of function"""
		return 0
	def general(self, num_steps=100, barch=50):
		"""Performs operation and completes parameters"""
		return


				
