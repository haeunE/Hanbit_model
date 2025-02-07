import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 데이터 컬럼 선택
features = ['lon', 'lat', 'year', 'month', 'day', 'hour'
            'no2', 'o3', 'co', 'so2', 'pm10', 'pm25', 'wd', 'ws',  
            'ta', 'td', 'hm', 'rn', 'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa',
             ]

# 데이터셋 로드
df = pd.read_csv("dust_data.csv", index_col=0)

# One-Hot Encoding (week)
df_week_encoded = pd.get_dummies(df['week'], prefix='week')
df = pd.concat([df.drop(columns=['week']), df_week_encoded], axis=1)

# One-Hot Encoding된 컬럼 추가
features += list(df_week_encoded.columns)  # week_0 ~ week_6 추가

# 입력(X)과 출력(Y) 데이터 분리
X = df[features].values
Y = df[['pm10', 'pm25']].values  # PM10, PM25 예측

# 데이터 중 훈련(80%), test(10%), validation(10%) 분리
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# LSTM 입력 형식 변환 (samples, time_steps, features)
time_steps = 24
X_train = X_train.reshape(-1, time_steps, len(features))
X_test = X_test.reshape(-1, time_steps, len(features))
X_val = X_val.values.reshape(-1, time_steps, len(features))

model = Sequential([
    TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'), input_shape=(time_steps, len(features))),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # PM2.5 예측
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse', metrics=['mae'])

# Early Stopping 추가 (과적합 방지)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32, callbacks=[early_stopping])

# 모델 평가
test_loss, test_mae = model.evaluate(X_test, Y_test)
print(f"Test MAE: {test_mae:.4f}")