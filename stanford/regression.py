import ml_common as ml

#error (not gradient)
def lms_error(training_set, hypothesis):
	total = 0;
	for i in range(training_set.size()):
		total += (hypothesis(training_set.get_input(i)) - training_set.get_output(i)) ** 2
	return total / (2 * training_set.size())

def linear(training_set, step_size, num_steps, batch_size=80, log_level=ml.NORMAL, log_frequency=100, plot_error=""):
	#TODO doesn't handly intercepts properly, padding ones doesn't work
	#training_set.pad_ones()
	params = [0] * training_set.get_num_features()

	error_plot = []

	hypothesis = lambda x: ml.mult_sum(params, x)
	for step in range(num_steps):
		for i in range(training_set.get_num_features()):
			#multiply by mean of correct feature
			grad = ml.gradient(hypothesis, training_set, i, batch_size)
			diff = (grad * step_size * ml.mean(training_set.get_plottable_input(i)))

			if step % log_frequency == 0:
				ml.log("Step " + str(step) + ": " + str(diff), ml.V_VERBOSE, log_level)

			params[i] -= diff

		hypothesis = lambda x: ml.mult_sum(params, x)
		if step % log_frequency == 0:
			ml.log("Error " + str(step) + ": " + str(lms_error(training_set, hypothesis)), ml.V_VERBOSE, log_level)
		if step % log_frequency == 0:
			ml.log("Hypothesis " + str(params), ml.V_VERBOSE, log_level)
		if plot_error != "":
			error_plot.append(lms_error(training_set, hypothesis))
		step += 1
	if plot_error != "":
		ml.plot_list(range(num_steps), error_plot, plot_error)	
	return hypothesis
