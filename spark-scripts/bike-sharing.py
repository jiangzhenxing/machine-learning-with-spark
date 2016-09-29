from pyspark.mllib.regression import LabeledPoint
import numpy as np

raw_data = sc.textFile('data/bike-sharing-dataset/hour_noheader.csv')
num_data = raw_data.count()
print(num_data)
records = raw_data.map(lambda line: line.split(','))
records.first()
mappings = [records.map(lambda record: record[i]).distinct().zipWithIndex().collectAsMap() for i in range(2,10)]

cat_len = sum(list(map(len, mappings)))
cat_len
num_len = len(records.first()[11:15])
num_len
total_len = cat_len + num_len

def extract_features(record):
	cat_vec = np.zeros(cat_len)
	i = 0   
	step = 0
	for field in record[2:9]:
		m = mappings[i]
		idx = m[field]
		cat_vec[step + idx] = 1
	    i += 1  
	    step += len(m)
	    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))

def extract_label(record):
	return float(record[-1])

def extract_features_dt(record):
	return np.array([float(x) for x in record[2:14]])

data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
first_point_dt = data_dt.first()

first_point_dt.label
first_point_dt.features
len(first_point_dt.features)

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree

linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
true_vs_predicted.take(5)

dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data_dt.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)
true_vs_predicted_dt.take(5)
dt_model.depth()
dt_model.numNodes()

def squared_error(actual, pred):
	return (pred - actual) ** 2
def abs_error(actual, pred):
    return np.abs(pred - actual)
def squared_log_error(actual, pred):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2

true_vs_predicted.map(lambda t: squared_error(t[0], t[1])).mean()
true_vs_predicted.map(lambda t: abs_error(t[0], t[1])).mean()
true_vs_predicted.map(lambda t: squared_log_error(t[0], t[1])).mean()


