import re
import numpy as np

def reader(txt_file):
    '''Reads a CSV with an arbitrary number of attributes
    and discrete values and formats the input for use with the
    NB_discrete function. The first row must contain the header
    or attribute names and the right-most column must contain the
    class values for the formatting to properly execute.

    For example (taken from NB_example_data.txt):

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
    are determined one simple rule: if state1 == state4, then
    the class is equal to the value of state1 and state4. Otherwise,
    the class value is 4. State2 and state3 are just noise.'''
    
    days = []
    with open(txt_file, 'r') as f:
        for row in f:
            m = re.split('\W*', row)
            temp_day = []
            for val in m:
                if val != '':
                    temp_day.append(val)
            days.append(temp_day)
            
    master = np.column_stack(days[1:])
    y = [val for val in master[-1]]
    x = master[:-1]
    return [x, y]
