import common as ml
import regression as r

ts = ml.TrainingSet(lambda x:x * 4, 30, -10, 10, 0)
reg = r.Regression(ts)
reg.general()
reg.plot(0)
ml.show_plot()
