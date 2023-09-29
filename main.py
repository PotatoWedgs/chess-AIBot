# Import dependencies
import torch 
import pandas as pd
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os

#   Our Chess dataset
data_frames = []

for i in range(500):
    file_path = f'chess_data/traincsv/game{i+1}.csv'
    if os.path.exists(file_path):
        data_frames.append(pd.read_csv(file_path))

data = pd.concat(data_frames)

for column in data.columns:
    if len(data.columns) > 97:
        if "col" in column:
            data = data.drop(column, axis=1)
            
    elif len(data.columns) == 97:
        break


#   Setuping out X and y data
X = data.drop('move_range_num', axis=1)
y = data['move_range_num']

X = np.array(X)
y = np.array(y)

#   Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)

#   Converting Train/Test data to Tensors   
X_train = torch.FloatTensor(X)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y)
y_test = torch.LongTensor(y_test)


# Chess Neural Network
class ChessNeuralNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(96, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 4096),
            nn.ReLU()
        )

    def forward(self, x): 
        return self.mod(x)


# Instance of the neural network, loss, optimizer 
model = ChessNeuralNetwork().cuda()
optimizer = Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss() 

# Training/Testing flow 
if __name__ == "__main__": 



    #   Training through 10000 epochs
    for i in range(100):


        #   Predicting and finding the loss
        y_pred = model.forward(X_train.cuda())
        loss = criterion(y_pred, y_train.cuda())


        #   Printing the loss
        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss}')


        #   Back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #   Testing
    with torch.no_grad():   #   Turning off back propogation

        correct = 0
        
        for iteration, tensor_data in enumerate(X_test):    
            y_val = model.forward(tensor_data.cuda())  #   Evaluating the test dataset

            if y_val.argmax().item() == y_test[iteration]:
                correct += 1
                print(f'{iteration+1}.) {str(y_val)},       {y_test[iteration]},    Correct')
            else:
                print(f'{iteration+1}.) {str(y_val)},       {y_test[iteration]},    Wrong')

        print(f'We got {correct}/{len(X_test)} correct')

    #torch.save(model.state_dict(), 'model.pt')