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
	def softmax(self, inputs, outcome=None):
		if (outcome != None and outcome < len(self.params)):
			explin = exp(1 * r.linear(self.params[outcome], inputs))
			return (explin / (explin + 1))
		else:
			for i in self.params:
				explin = exp(1 * r.linear(i, inputs))
				return (explin / (explin + 1))
	def update_rule(self, feature, class_, examples):
		"""Feature i class j update rule for examples in training set examples	
		return float to change by"""
		grad =  0
		for i in examples:
			#TODO add weight
			diff = (1 if (self.ts.get_output(i) == class_) else 0) - regression_function(self.ts.get_input(i), class_)
			grad += self.ts.get_input_feature(i, feature) * 
		return (grad/max(len(examples), 1))

	def __init__(self, train):
		#logistic is technically a subclass of softmax
		super().__init__(train)
		self.regression_function = self.softmax
		self.step_size = .01
	def error(self):
		"""Return error of function"""
		return 0
	def general(self, num_steps=100, barch=50):
		"""Performs operation and completes parameters"""
		return
	@staticmethod
	def test():
		return 0	


				
