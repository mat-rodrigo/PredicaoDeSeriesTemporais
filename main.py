import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# 1. Carregar os dados
data = np.sin(np.arange(0, 100, 0.1))
df = pd.DataFrame(data, columns=['Close'])

# 2. Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# 3. Preparar os dados para o modelo LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(df_scaled, time_step)

# Redimensionar para o formato [samples, time steps, features] exigido pelo LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. Construir o modelo LSTM
model = Sequential()
model.add(Input(shape=(time_step, 1)))  # Usando a camada Input explicitamente
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Treinar o modelo
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# 6. Fazer previsões
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)

# 7. Visualizar os resultados
plt.plot(scaler.inverse_transform(df_scaled), label='Real Data')
plt.plot(train_predict, label='LSTM Predictions')
plt.legend()
plt.show()
