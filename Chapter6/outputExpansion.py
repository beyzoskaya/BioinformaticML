import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

data_path = '/Users/beyzakaya/Desktop/bk/Proje:Kod/Chapter3/Chapter6/datasets/JPY=X.csv'
df = pd.read_csv(data_path)
#print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#print(df.head())
df.dropna(inplace=True)

# close prices for prediction
#prices = df['Close'].values
#print(f"Shape of prices: {prices}")
prices = df['Close'].values.reshape(-1, 1)
#print(f"Shape of prices after reshape: {prices}")

scaler = MinMaxScaler()
normalized_prices = scaler.fit_transform(prices)
#print(f" Normalized prices: {normalized_prices}")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('USD/JPY Close Price Over Time')
plt.savefig('usd_jpy_close_price.png')
#plt.show()

# Data prep for output expansion
def prepare_data(prices, window_size=30, output_days=5):
    X, y = [], []
    for i in range(len(prices)- window_size-output_days):
        #print(f"Shape of prices window: {prices[i:i+window_size].shape}")
        #print(f"Shape of prices window after flatten: {prices[i:i+window_size].flatten().shape}")
        X.append(prices[i:i+window_size].flatten())
        y.append(prices[i + window_size:i + window_size + output_days].flatten())
        """
        X[0] prices[0:30] contains prices from day 0 to day 29
        y[0] prices[30:35] contains prices from day 30 to 34
        """
    return np.array(X), np.array(y)

output_days = 5
window_size =30
train_size = int(len(normalized_prices) * 0.8)
train_prices = normalized_prices[:train_size]
test_prices = normalized_prices[train_size:]

train_X, train_y = prepare_data(train_prices,window_size,output_days)
test_X, test_y = prepare_data(test_prices,window_size,output_days)

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

class CurrencyPredictionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CurrencyPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

def train_model(train_X, train_y, test_X, test_y, epochs=50):
    input_dim = train_X.shape[1]
    hidden_dim = 128
    output_dim = output_days

    model = CurrencyPredictionNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in zip(train_X, train_y):
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # test loss is nan so gradient clipping is used
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_X):.4f}')

    model.eval()
    with torch.no_grad():
        predictions = model(test_X)
        print(f"Predictions: {predictions}")
        print(f"Predictions contain NaNs: {torch.isnan(predictions).any()}")
        test_loss = criterion(predictions, test_y)
        print(f"Test targets contain NaNs: {torch.isnan(test_y).any()}")
        print(f'Test Loss: {test_loss.item():.4f}')
       
        plt.figure(figsize=(14, 7))
        plt.plot(range(output_days), test_y[0].numpy(), label='Actual')
        plt.plot(range(output_days), predictions[0].numpy(), label='Predicted')
        plt.xlabel('Days')
        plt.ylabel('Normalized Close Price')
        plt.title('USD/JPY Close Price Prediction for Next 5 Days')
        plt.legend()
        plt.savefig('close_price_predict_output_expansion.png')
        plt.show()

train_model(train_X, train_y, test_X, test_y, epochs=50)