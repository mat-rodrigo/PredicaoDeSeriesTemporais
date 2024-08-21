# Predição De Series Temporais
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
"numpy" e "pandas" são usados para manipulação de dados e cálculos numéricos;  
"matplotlib.pyplot" utilizado para visualizar os dados e previsões;  
"MinMaxScaler" da "sklearn.preprocessing" usado para normalizar os dados entre 0 e 1;  
"Sequential", "LSTM", "Dense", "Dropout", "Input" da "Keras" usados para construir e treinar o modelo de rede neural.

## 2. Leitura e Preparação dos Dados

```
data = np.sin(np.arange(0, 100, 0.1))
df = pd.DataFrame(data, columns=['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
```
Geramos uma série temporal baseada na função seno, com valores de 0 a 100 em incrementos de 0,1;  
Após isto convertemos a série temporal em um DataFrame do pandas, com uma coluna chamada 'Close'. Esta simula os dados que queremos prever (por exemplo, o preço de fechamento de uma ação);  
Criamos um objeto scaler que normaliza os dados para o intervalo [0, 1];    
E aplicamos a normalização nos dados do DataFrame df, retornando uma matriz numpy escalad.

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
Esta função transforma a série temporal em pares de entrada-saída para treinar o LSTM. Onde "time_step" Define o número de passos anteriores usados para prever o próximo valor.
Para cada i, o intervalo "i:(i + time_step)" é usado como entrada (X), e o valor em "i + time_step" como saída (Y).
Retornando X e Y como arrays numpy.  
Para criar o Dataset usamos o "time_step = 10" para definir o número de passos (janelas de tempo) que serão usados para fazer a previsão e X, y = create_dataset(df_scaled, time_step): Usa a função create_dataset para criar os arrays X e y a partir dos dados normalizados.

## 4. Redimensionar para o formato [samples, time steps, features] exigido pelo LSTM

```
X = X.reshape(X.shape[0], X.shape[1], 1)
```
O LSTM espera os dados no formato [samples, time steps, features].  
Para isso X.reshape(X.shape[0], X.shape[1], 1) Redimensiona X para que cada janela de tempo (X.shape[1]) tenha apenas uma característica (a coluna de dados).

## 5. Construir o modelo LSTM
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
Aqui Inicializamos um modelo sequencial;  
Definimos a forma da entrada como (time_step, 1), onde time_step é o número de passos temporais e 1 é o número de características;  
Adiciona uma camada LSTM com 50 unidades. E return_sequences=True significa que a saída de cada LSTM será usada como entrada para a próxima camada LSTM;  
Adicionamos uma camada de dropout com uma taxa de 20%, que ajuda a evitar overfitting, descartando 20% dos neurônios de forma aleatória;  
Deposi adicionamos outra camada LSTM com 50 unidades, desta vez sem retornar a sequência completa (apenas a última saída é retornada).
Também adicionamos uma camada densa totalmente conectada com 1 unidade, que é a previsão final.  
  
Na ultima linha fazemos a compilação do modelo onde usamos o otimizador Adam, que é eficiente para redes neurais profundas e
loss='mean_squared_error' Define a função de perda como o erro quadrático médio, que é apropriado para problemas de regressão.

## 6. Treinamento do modelo

```
model.fit(X, y, epochs=100, batch_size=32, verbose=1)
```
O treinamento de modelo fizemos usando X como entrada e y como saída por 100 épocas, com um batch size de 32. verbose=1 mostra o progresso do treinamento

## 7. Fazendo previsões

```
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
```
Para fazer previsões e inverter a normalização usamos model.predict(X) para o modelo treinado fazer previsões usando os mesmos dados de entrada X.
E scaler.inverse_transform(train_predict) reverte a normalização para que as previsões estejam na mesma escala dos dados originais.

## 8. Visualização dos resultados

```
plt.plot(scaler.inverse_transform(df_scaled), label='Real Data')
plt.plot(train_predict, label='LSTM Predictions')
plt.legend()
plt.show()
```
Na visualização dos resultados  
plt.plot(scaler.inverse_transform(df_scaled), label='Real Data'): Plota os dados reais, revertendo a normalização.  
plt.plot(train_predict, label='LSTM Predictions'): Plota as previsões feitas pelo modelo.  
plt.legend(): Adiciona uma legenda ao gráfico para diferenciar os dados reais das previsões.  
plt.show(): Exibe o gráfico.
