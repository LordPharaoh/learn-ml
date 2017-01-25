import regression as r
def logistic(list1, list2):
	"""Logistic hp(x), do the linear thing and then exp that. (exp(lin)) / (exp(lin)+1)"""
	explin = exp(1 * linear(list1, list2))
	return (explin / (explin + 1))
