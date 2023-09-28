# Import dependencies
import torch 
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pathlib import Path
from functools import lru_cache

"""#   Our Iris dataset
data = pd.read_csv('chess_data/csv/game1.csv')

#   Setuping out X and y data
X = data.drop('move_range_num', axis=1)
y = data['move_range_num']

X = np.array(X)
y = np.array(y)

#   Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)

#   Converting Train/Test data to Tensors   
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)"""

@lru_cache()
def get_sample_count_by_file(path: Path) -> int:
    c = 0
    with path.open() as f:
        for line in f:
            c += 1
    return c


class CSVDataset:
    def __init__(self, csv_directory: str, extension: str = ".csv"):
        self.directory = Path(csv_directory)
        self.files = sorted((f, get_sample_count_by_file(f)) for f in self.directory.iterdir() if f.suffix == extension)
        self._sample_count = sum(f[-1] for f in self.files)

    def __len__(self):
        return self._sample_count

    def __getitem__(self, idx):
        current_count = 0

        history_window = 2
        my_idx=idx+2

        for file_, sample_count in self.files:
            if current_count <= my_idx < current_count + sample_count:
                break  
            current_count += sample_count

        file_idx = my_idx - current_count # the index we want to access in file_
        if file_idx < 2:
            file_idx += 2

        with file_.open() as f:
            data = []
            for i, line in enumerate(f):
                if i >= file_idx-history_window and i <= file_idx:
                    for v in line.split(","):
                        data.append(float(v))

            data = np.array(data)
            return torch.from_numpy(data)

dataset = CSVDataset("chess_data/csv")
loader = DataLoader(dataset, batch_size=1)

print(list(enumerate(loader)))

"""
# Chess Neural Network
class ChessNeuralNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(96, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 4096),
            nn.Softmax()
        )

    def forward(self, x): 
        return self.mod(x)


# Instance of the neural network, loss, optimizer 
model = ChessNeuralNetwork().cuda()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() 

# Training/Testing flow 
if __name__ == "__main__": 



    #   Training through 10000 epochs
    for i in range(10000):


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

    #torch.save(model.state_dict(), 'model.pt')"""