import NB_discrete as NB
import NB_reader

input_data = NB_reader.reader('NB_example_data.txt')

x = input_data[0]
y = input_data[1]
obs = ['2','3','2','3']

pred = NB.predict(x,y,obs)

print 'Predicted class:', pred[0]
print pred[1]
