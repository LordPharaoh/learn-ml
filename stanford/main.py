import ml_common as ml
import regression
#TODO works but sometimes the whole training set is 0'd and error goes to inifinity?
test_func = lambda x, y: (3 * x + y)
ts = ml.generate_training_set(test_func, 80, -10, 10)
ts.plot(0)
ml.plot_hypothesis(regression.linear(ts, .01, 1000, log_level=ml.V_VERBOSE, log_frequency=100, plot_error="b-"), -100, 100)
#ml.plot_function(test_func, -10, 10, 1, "g-")
ml.show_plot()
