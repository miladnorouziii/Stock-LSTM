import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch 
import torch.nn as nn
from torch.autograd import Variable
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

data = pd.read_csv('Dataset/SBUX.csv')

#print(data.shape)
#print(data.head())
plt.title("Starbucks Stock Volume")
plt.plot(data['Volume'])
#plt.show()

X = data[['Open','High','Low','Close','Adj Close']]
Y = data['Volume']
Y = Y.values.reshape(Y.shape[0], 1)

#print(X)
print("-----------------------------------")
#print(Y)

ss = StandardScaler()
mm = MinMaxScaler()

X_SS = ss.fit_transform(X)
Y_mm = mm.fit_transform(Y)
print(X_SS.shape)

#Set test and Train Validation dataset
X_Train = X_SS[:200, :]
X_Test = X_SS[200:, :]
Y_Train = Y_mm[:200, :]
Y_Test = Y_mm[200:, :]

X_Train_Tensor = Variable(torch.Tensor(X_Train))
X_Test_Tensor = Variable(torch.Tensor(X_Test))
Y_Train_Tensor = Variable(torch.Tensor(Y_Train))
Y_Test_Tensor = Variable(torch.Tensor(Y_Test))

X_Train_Tensor_Final = torch.reshape(X_Train_Tensor, (X_Train_Tensor.shape[0], 1, X_Train_Tensor.shape[1]))
X_Test_Tensor_Final = torch.reshape(X_Test_Tensor, (X_Test_Tensor.shape[0], 1, X_Test_Tensor.shape[1]))

print("Train data shape ->", X_Train_Tensor_Final.shape, Y_Train.shape)
print("Test data shape ->", X_Test_Tensor_Final.shape, Y_Test.shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, seq_length):
        super(LSTM, self).__init__()
        self.num_layer = num_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size= input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layer, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layer, x.size(0), self.hidden_size))
        output, (hn,cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)

        output = output[:,-1,:]
        output = self.relu(output)
        output = self.fc_1(output)
        output = self.relu(output)
        output = self.fc(output)

        return output
    
num_epochs = 1000
learning_rate = 0.005
input_size = 5
hidden_size = 2
num_layer = 1
lstm = LSTM(input_size, hidden_size, num_layer, X_Train_Tensor_Final.shape[1])
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = lstm.forward(X_Train_Tensor_Final)
    loss = loss_func(output, Y_Train_Tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" %(epoch, loss.item()))

df_X_ss = ss.transform(data[['Open','High','Low','Close','Adj Close']]) #old transformers
df_y_mm = mm.transform(data.iloc[:, -1:]) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm(df_X_ss)#forward pass
data_predict = train_predict.data.numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=200, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 
