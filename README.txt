    Use Bayes' Rule, assuming discrete, independent predictors,
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