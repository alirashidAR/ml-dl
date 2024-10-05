import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

# Load the data
url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
data = pd.read_csv(url)
print(data.shape)
data =  data.dropna()

train_input =  np.array([data.x[0:500]]).reshape(500,1)
train_output = np.array([data.y[0:500]]).reshape(500,1)

test_input = np.array([data.x[500:700]]).reshape(199,1)
test_output = np.array([data.y[500:700]]).reshape(199,1)

class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def forward_propogation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions =  np.multiply(m,train_input) + c
        return predictions
    
    def cost_function(self,predictions,train_output):
        cost = np.mean((train_output-predictions)**2)
        return cost
    
    def backward_propogation(self,train_input,train_output,predictions):
        derivatives = {}
        df = predictions-train_output
        dm = 2*np.mean(np.multiply(train_input,df))
        dc = 2* np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc

        return derivatives
    
    def update_parameters(self,derivatives,learning_rate):
        self.parameters['m'] = self.parameters['m'] -  learning_rate*derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate*derivatives['dc']

    def train(self,train_input,train_output,learning_rate,iters):
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

       
        for i in range(iters):
            self.loss = []
            predictions = self.forward_propogation(train_input) 

                
            cost = self.cost_function(predictions, train_output) 

            derivatives = self.backward_propogation( 
                    train_input, train_output, predictions) 

            self.loss.append(cost)
            self.update_parameters(derivatives, learning_rate) 
            print("Coefficient of x: " + str(self.parameters['m']))
            print("Loss of x: "+ str(self.loss[-1]))

        
        return self.parameters, self.loss 
    

linear_reg =  LinearRegression()
parameters, loss =  linear_reg.train(train_input,train_output,0.0001,20)
