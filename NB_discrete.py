from __future__ import division
import numpy as np
import heapq


def fit_model(x, y):

    laplace_estimator = 1
    attributes = [str(i) for i in range(0,len(x)+1)]      
    uniq_targets = set(y)
    
    if laplace_estimator==1:
        target_laplace = len(uniq_targets)
        laplace_one = 1
    else:
        target_laplace = 0
        laplace_one = 0
        
    total = len(y)+target_laplace

    class_targets = {}
    class_targets_targetname = {}
    class_counts = {}
    class_names = []

    i=0
    for target in uniq_targets:
        class_targets[i] = target
        class_targets_targetname['_class'+str(i)] = target
        class_counts[i] = y.count(target)
        class_names.append('_class'+str(i))
        i+=1
    
    class_values = {}
    freq = {}

    i=0
    for attrib_vec in x:
        for k in range(0, len(class_targets)):
            class_values[k] = []
        j=0
        for val in attrib_vec:
            for k in range(0, len(class_targets)):
                if y[j]==class_targets[k]:
                    class_values[k].append(val)
            j+=1
        for k in range(0, len(class_targets)):
            freq[attributes[i]+'_class'+str(k)] = []
            uniq_vals = set(attrib_vec)
            for val in uniq_vals:
                if laplace_estimator==1:
                    laplace = len(uniq_vals)
                else:
                    laplace = 0
                freq[attributes[i]+'_class'+str(k)].append({val:(class_values[k].count(val)+laplace_one)/(class_counts[k]+laplace)})
        i+=1
    attributes = attributes[:-1]
    
    return [class_names, class_targets_targetname, attributes, freq, class_counts, total, laplace_one]


def predict(x, y, new_obs):
    '''Use Bayes' Rule, assuming discrete, independent predictors,
    to estimate the probability that a new observation belongs to
    a certain class. A Laplace Estimator is used to handle cases
    where the predictor does not occur in the training data, which
    would otherwise result in a zero probability estimate for that
    predictor. The Laplace Estimator can be toggled on or off within
    the fit_model function.

    Parameters:
    x is array-like with shape = [n_attributes, n_values]
    y is array-like with shape = [n_values]
    new_obs is array-like with shape = [n_values]

    Output:
    The output is a list with index 0 equal to the class with the
    highest probability and index 1 equal to a list of lists containing
    class-probability pairs.

    Example:



import NB_discrete as NB
import numpy as np

raw_data = [['state1', 'state2', 'state3', 'state4', 'class'], ['1', '2', '2', '1', '1'],
['2', '1', '2', '3', '4'], ['2', '3', '2', '2', '2'], ['2', '3', '3', '3', '4'],
['3', '1', '1', '3', '3'], ['2', '2', '2', '2', '2'], ['1', '3', '2', '1', '1'],
['2', '3', '3', '2', '2'], ['1', '1', '3', '1', '1'], ['1', '3', '3', '3', '4'],
['3', '2', '2', '1', '4'], ['1', '2', '3', '2', '4'], ['3', '2', '2', '3', '3'],
['3', '3', '1', '2', '4'], ['2', '1', '1', '3', '4'], ['3', '3', '3', '3', '3'],
['1', '2', '3', '3', '1'], ['2', '2', '3', '2', '2'], ['3', '3', '1', '3', '3']]

x = np.column_stack(raw_data[1:]) # remove header
y = [val for val in x[-1]] # store class values in y
x = x[:-1] # remove class values from x
new_obs = ['2','3','2','2']

pred = NB.predict(x,y,new_obs)

print pred[0]
# prints 2

print pred[1]
# prints [['1', 0.020525518967221563], ['3', 0.020525518967221563],['2', 0.76970696127080884],
# ['4', 0.18924200079474812]]



    For background on raw_data above, see the table and explanation below.

    state1, state2, state3, state4, class
    1,      2,      2,      1,      1
    2,      1,      2,      3,      4
    2,      3,      2,      2,      2
    2,      3,      3,      3,      4
    3,      1,      1,      3,      3
    2,      2,      2,      2,      2
    1,      3,      2,      1,      1
    2,      3,      3,      2,      2
    1,      1,      3,      1,      1
    1,      3,      3,      3,      4
    3,      2,      2,      1,      4
    1,      2,      3,      2,      4
    3,      2,      2,      3,      3
    3,      3,      1,      2,      4
    2,      1,      1,      3,      4
    3,      3,      3,      3,      3
    1,      2,      3,      3,      1
    2,      2,      3,      2,      2
    3,      3,      1,      3,      3

    In the above table, each state can take values 1, 2, or 3
    and the class can take values 1, 2, 3, or 4.The class values
    are determined one rule: if state1 == state4, then
    the class is equal to the value of state1 and state4. Otherwise,
    the class value is 4. State2 and state3 are just noise.

    data = fit_model(x, y)
    class_names = data[0]
    class_targets_targetname = data[1]
    attributes = data[2]
    freq = data[3]
    class_counts = data[4]
    total = data[5]
    laplace_one = data[6]
    
    class_probs = {}
    sum_probs = []
    j=0
    for class_name in class_names:
        class_probs[class_name] = []
        i=0
        for attrib in attributes:
            for key in freq[attrib+class_name]:
                if key.keys()[0]==new_obs[i]:
                    class_probs[class_name].append(key.values())
            i+=1
        class_probs[class_name]=np.product(class_probs[class_name])*(class_counts[j]+laplace_one)/total
        sum_probs.append(class_probs[class_name])
        j+=1

    output = []
    for class_name in class_names:
        output.append([class_targets_targetname[class_name], class_probs[class_name]/sum(sum_probs)])
    predicted_class = class_targets_targetname[heapq.nlargest(1,class_probs.items(), key=lambda(k,v):v)[0][0]]

    return [predicted_class, output]





