# PredicaoDeSeriesTemporais
Uma técnica utilizada para prever valores futuros com base em dados históricos organizados em uma sequência temporal. Essas previsões são amplamente utilizadas em diversas áreas, como finanças (previsão de preços de ações), economia (previsão de PIB), climatologia (previsão do tempo), entre outras.


# Previsão de Consumo de Energia com XGBoost

Repositório criado para disponibilizar os arquivos referente á previsão de consumo de energia com Python usando um modelo de machine learning XGBoost.

## 1. Importação de Bibliotecas

```shell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
```

## 2. Leitura e Preparação dos Dados

```
data = np.sin(np.arange(0, 100, 0.1))
df = pd.DataFrame(data, columns=['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
```


## 3. Preparação dos dados para o modelo LSTM

```
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(df_scaled, time_step)
```

## 4. Redimensionar para o formato [samples, time steps, features] exigido pelo LSTM

```
X = X.reshape(X.shape[0], X.shape[1], 1)
```

## 5. Construindo o modelo LSTM

```
model = Sequential()
model.add(Input(shape=(time_step, 1)))  # Usando a camada Input explicitamente
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
```

## 6. Treinamento do modelo

```
model.fit(X, y, epochs=100, batch_size=32, verbose=1)
```

## 7. Fazendo previsões

```
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
```

## 8. Visualização dos resultados

```
plt.plot(scaler.inverse_transform(df_scaled), label='Real Data')
plt.plot(train_predict, label='LSTM Predictions')
plt.legend()
plt.show()
```
