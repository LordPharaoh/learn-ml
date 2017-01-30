import regression as r
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
			
