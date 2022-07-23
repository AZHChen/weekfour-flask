# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('insurance.csv')
#dataset = pd.get_dummies(dataset,drop_first=True)

dataset = dataset.drop('region',axis=1)
X = dataset.drop('charges',axis=1)

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'female':0, 'male':1, 'no':0, 'yes':1, 'southwest':0,
                 'southeast':1, 'northeast':2, 'northwest':3}
    return word_dict[word]

X['sex'] = X['sex'].apply(lambda x : convert_to_int(x))
X['smoker'] = X['smoker'].apply(lambda x : convert_to_int(x))
y = dataset['charges']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[22, 1, 21,0,0]]))