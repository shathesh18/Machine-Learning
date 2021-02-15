from typing import Dict, Any
from utils import *
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    positive_rate_dict = {}
    Accuracy_measures = {}
    group_keys = list(categorical_results.keys())
    for key in group_keys:
        temp = categorical_results[key]
        PR, PPR,group, correct_array, total_array, accuracy_array = ROC_positive_rate(temp, key)
        positive_rate_dict[key] = PR
        Accuracy_measures[key] = [correct_array, total_array, accuracy_array]
    final_threshold = getFinalThresholds(group_keys, epsilon, positive_rate_dict, Accuracy_measures)
    if len(final_threshold) == 0:
        print('Demographic Parity cannot be enforced')
        return None, None
    for i, key in enumerate(group_keys):
        thresholds[key] = (final_threshold[i] + 1) / 100
    for i, key in enumerate(group_keys):
        temp = categorical_results[key]
        demographic_parity_data[key] = apply_threshold(temp, (final_threshold[i] + 1) / 100)
    return demographic_parity_data, thresholds



#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}
    True_positive_dict={}
    Accuracy_measures={}
    group_keys=list(categorical_results.keys())
    for key in group_keys:
        temp = categorical_results[key]
        f_scores, TPR, FPR, group,correct_array,total_array,accuracy_array = get_ROC_Equal_Opportunity(temp, key)
        True_positive_dict[key] = TPR
        Accuracy_measures[key] = [correct_array, total_array, accuracy_array]
    final_threshold=getFinalThresholds(group_keys,epsilon,True_positive_dict,Accuracy_measures)
    if len(final_threshold) == 0:
        print('Equal Opportunity cannot be enforced')
        return None, None
    for i,key in enumerate(group_keys):
        thresholds[key]=(final_threshold[i]+1)/100
    for i,key in enumerate(group_keys):
        temp=categorical_results[key]
        equal_opportunity_data[key]=apply_threshold(temp,(final_threshold[i]+1)/100)
    return  equal_opportunity_data,thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    group_keys=list(categorical_results.keys())
    for key in group_keys:
        thresholds[key]=get_best_accuracy(categorical_results[key])
        mp_data[key]=apply_threshold(categorical_results[key],thresholds[key])
    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    predictive_positive_rate_dict = {}
    Accuracy_measures = {}
    group_keys = list(categorical_results.keys())
    for key in group_keys:
        temp = categorical_results[key]
        PR, PPR, group, correct_array, total_array, accuracy_array = ROC_positive_rate(temp, key)
        predictive_positive_rate_dict[key] = PPR
        Accuracy_measures[key] = [correct_array, total_array, accuracy_array]
    final_threshold = getFinalThresholds(group_keys, epsilon, predictive_positive_rate_dict,Accuracy_measures)
    if len(final_threshold) == 0:
        print('Predictive Parity cannot be enforced')
        return None,None
    for i, key in enumerate(group_keys):
        thresholds[key] = (final_threshold[i] + 1) / 100
    for i, key in enumerate(group_keys):
        temp = categorical_results[key]
        predictive_parity_data[key] = apply_threshold(temp, (final_threshold[i] + 1) / 100)
    return predictive_parity_data, thresholds



    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}
    group_keys=list(categorical_results.keys())
    temp_accuracy=0
    temp_theshold=0
    for i in range(1, 101):
        temp_single_threshold_data = {}
        threshold = float(i) / 100.0
        for key in group_keys:
            eval_copy = list.copy(categorical_results[key])
            eval_copy = apply_threshold(eval_copy, threshold)
            temp_single_threshold_data[key]=eval_copy
        accuracy=get_total_accuracy(temp_single_threshold_data)
        if(accuracy>temp_accuracy):
            temp_accuracy=accuracy
            temp_theshold=threshold
            single_threshold_data=temp_single_threshold_data
    for i,key in enumerate(group_keys):
        thresholds[key]=temp_theshold
    return single_threshold_data, thresholds

#######################################################################################################################
#custom codes
#######################################################################################################################

#######################################################################################################################

def get_ROC_Equal_Opportunity(prediction_label_pairs, group):
    true_positives = []
    false_positives = []
    f_scores=[]
    num_correct_array=[]
    total_num_array=[]
    local_accuracy_array=[]
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(prediction_label_pairs)
        eval_copy = apply_threshold(eval_copy, threshold)
        TPR = get_true_positive_rate(eval_copy)
        FPR = get_false_positive_rate(eval_copy)
        num_correct,total_num,localAccuracy=get_local_accuracy(eval_copy)
        f_score=get_num_predicted_positives(eval_copy)/len(eval_copy)
        true_positives.append(TPR)
        false_positives.append(FPR)
        f_scores.append(f_score)
        num_correct_array.append(num_correct)
        total_num_array.append(total_num)
        local_accuracy_array.append(localAccuracy)

    return (f_scores,true_positives, false_positives, group,num_correct_array,total_num_array,local_accuracy_array)

#######################################################################################################################


#######################################################################################################################

#checks for the equals with epsilon condition
def checkEqual(x, y, upperLimit, lowerLimit, limit):
    if (y <= x + upperLimit and y >= x + lowerLimit):
        difference = y - x
        if difference > 0:
            if lowerLimit is -limit:
                return True, upperLimit, difference - limit
            else:
                return True, upperLimit, max(lowerLimit, difference - limit)
        if difference < 0:
            if upperLimit is limit:
                return True, difference + limit, lowerLimit
            else:
                return True, min(upperLimit, difference + limit), lowerLimit
    return False, upperLimit, lowerLimit

#######################################################################################################################


#######################################################################################################################

#loopcheck: loops the two arrays and checks equal pairs
def loopCheck(x, y, upperLimit, lowerLimit, x_accuracy_parameters, y_accuracy_parameters):
    temp=[]
    if upperLimit is None and lowerLimit is None:
        for i in range (len(x)):
            k = x[i][5]
            for j in range (len(y)):
                con,uL,lL=checkEqual(x[i][2],y[j],x[i][3],x[i][4],.01)
                if con:
                    eval_copy=list.copy(k)
                    eval_copy.append(j)
                    correctx = x[i][6]
                    correcty = y_accuracy_parameters[0][j]
                    num_correct = correctx + correcty
                    totalx = x[i][7]
                    totaly = y_accuracy_parameters[1][j]
                    total = totalx + totaly
                    temp.append([x[i][0], j, x[i][2], uL, lL,eval_copy,num_correct,total,num_correct/total])
        return temp
    else:
        for i in range (len(x)):
            for j in range (len(y)):
                con,uL,lL=checkEqual(x[i], y[j], upperLimit, lowerLimit, .01)
                if(con):
                    k=[i,j]
                    correctx=x_accuracy_parameters[0][i]
                    correcty=y_accuracy_parameters[0][j]
                    num_correct=correctx+correcty
                    totalx=x_accuracy_parameters[1][i]
                    totaly=y_accuracy_parameters[1][j]
                    total=totalx+totaly
                    temp.append([i,j,x[i],uL,lL,k,num_correct,total,num_correct/total])
        return temp

#######################################################################################################################


#######################################################################################################################

#extracts the final thresholds
def getFinalThresholds(groupKeys,epsilon,metrics_dict,Accuracy_measures):
    if len(groupKeys) == 1:
        return "only one demographic group. Equal opportunity not needed to be enforced"
    x = metrics_dict[groupKeys[0]]
    y = metrics_dict[groupKeys[1]]
    ax = Accuracy_measures[groupKeys[0]]
    ay = Accuracy_measures[groupKeys[1]]
    result_store = loopCheck(x, y, epsilon, -epsilon,ax,ay)
    print('This is the length yu look for',len(result_store))
    if(result_store.__contains__([.16,.11,.07,.04])):
        print(result_store)
    else:
        print('get lost')
    if len(groupKeys) > 2:
        for i in range(2, len(groupKeys)):
            result_store = loopCheck(result_store, metrics_dict[groupKeys[i]], None, None,None,Accuracy_measures[groupKeys[i]])
    tempacc = 0
    finalThresholds=[]
    for ten in result_store:
        thresholds = ten[5]
        tAc = ten[-1]
        if tAc > tempacc:
            tempacc = tAc
            finalThresholds = list.copy(thresholds)
    return finalThresholds

#######################################################################################################################


#######################################################################################################################

#ROC data for demographic parity and predictive parity
def ROC_positive_rate(prediction_label_pairs, group):
    positives = []
    num_correct_array=[]
    total_num_array=[]
    local_accuracy_array=[]
    positive_predictive_array=[]
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(prediction_label_pairs)
        eval_copy = apply_threshold(eval_copy, threshold)
        PR = get_num_predicted_positives(eval_copy)
        PPR=get_positive_predictive_value(eval_copy)
        positive_predictive_array.append(PPR)
        num_correct,total_num,localAccuracy=get_local_accuracy(eval_copy)
        positives.append(PR/total_num)
        num_correct_array.append(num_correct)
        total_num_array.append(total_num)
        local_accuracy_array.append(localAccuracy)

    return (positives,positive_predictive_array,group,num_correct_array,total_num_array,local_accuracy_array)


#######################################################################################################################


#######################################################################################################################

#for getting the local financials
def get_best_cost(data):
    # Costs for the various categories
    tp_val = -60076
    tn_val = 23088
    fp_val = -110076
    fn_val = -202330
    profit=-inf
    best_threshold=0
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(data)
        eval_copy = apply_threshold(eval_copy, threshold)
        num_tp = get_num_true_positives(eval_copy)
        num_tn = get_num_true_negatives(eval_copy)
        num_fp = get_num_false_positives(eval_copy)
        num_fn = get_num_false_negatives(eval_copy)
        total = 0.0
        total += num_tp * tp_val
        total += num_tn * tn_val
        total += num_fp * fp_val
        total += num_fn * fn_val
        if(total>profit):
            profit=total
            best_threshold=threshold

    return best_threshold

#######################################################################################################################

#######################################################################################################################

#for getting the local accuracies
def get_best_accuracy(data):
    accuracy=0
    best_threshold=0
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(data)
        eval_copy = apply_threshold(eval_copy, threshold)
        temp_accuracy=get_local_accuracy(eval_copy)[-1]

        if(temp_accuracy>accuracy):
            accuracy=temp_accuracy
            best_threshold=threshold

    return best_threshold

#######################################################################################################################


#######################################################################################################################
#Method to get local accuracy of a certain demographic group

def get_local_accuracy(classification):
    total_correct = 0.0
    total_num_cases = 0.0


    for prediction, label in classification:
        total_num_cases += 1.0
        if prediction == label:
            total_correct += 1.0

    return total_correct , total_num_cases,total_correct/total_num_cases

#######################################################################################################################
