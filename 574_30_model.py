from sklearn import svm
from Postprocessing import *
from Preprocessing import preprocess
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

SVR=svm.SVR(kernel='rbf',gamma=.10,max_iter=5000)
SVR.fit(training_data, training_labels)

training_class_predictions = SVR.predict(training_data)
training_predictions = []
test_class_predictions = SVR.predict(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, .01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

print("")
print("Chosen Post Processing Technique - Equal Opportunity")
print("Chosen Secondary Optimization - Accuracy")
print("")
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")


print("Test data threshold")
for group in test_race_cases.keys():
    print("Threshold for " + group + ": " + str(thresholds[group]))
print("")

print("Accuracy on Test data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on Test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")
for group in training_race_cases.keys():
    accuracy = get_num_correct(training_race_cases[group]) / len(training_race_cases[group])
    print("Accuracy for " + group + " in training data: " + str(accuracy))
print("")

for group in test_race_cases.keys():
    accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
    print("Accuracy for " + group + ": in test data " + str(accuracy))


print("")
for group in training_race_cases.keys():
    FPR = get_false_positive_rate(training_race_cases[group])
    print("FPR for " + group + " in training data: " + str(FPR))

print("")
for group in test_race_cases.keys():
    FPR = get_false_positive_rate(test_race_cases[group])
    print("FPR for " + group + "in test data: " + str(FPR))

print("")
for group in training_race_cases.keys():
    FNR = get_false_negative_rate(training_race_cases[group])
    print("FNR for " + group + " in training data: " + str(FNR))

print("")
for group in test_race_cases.keys():
    FNR = get_false_negative_rate(test_race_cases[group])
    print("FNR for " + group + " in test data: " + str(FNR))

print("")
for group in training_race_cases.keys():
    TPR = get_true_positive_rate(training_race_cases[group])
    print("TPR for " + group + " in training data: " + str(TPR))

print("")
for group in test_race_cases.keys():
    TPR = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + " in test data: " + str(TPR))


print("")
for group in training_race_cases.keys():
    TNR = get_true_negative_rate(training_race_cases[group])
    print("TNR for " + group + " in training data: " + str(TNR))

print("")
for group in test_race_cases.keys():
    TNR = get_true_negative_rate(test_race_cases[group])
    print("TNR for " + group + " in test data: " + str(TNR))

print("")
for group in training_race_cases.keys():
    PPV = get_positive_predictive_value(training_race_cases[group])
    print("PPV for " + group + " in training data: " + str(PPV))

print("")
for group in test_race_cases.keys():
    PPV = get_positive_predictive_value(test_race_cases[group])
    print("PPV for " + group + " in test data: " + str(PPV))

print("")
for group in training_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(training_race_cases[group])
    prob = num_positive_predictions / len(training_race_cases[group])
    print("Probability of positive prediction for " + str(group) + " in training data: " + str(prob))

print("")
for group in test_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
    prob = num_positive_predictions / len(test_race_cases[group])
    print("Probability of positive prediction for " + str(group) + " in test data: " + str(prob))

print("")
