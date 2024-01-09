import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('heart.csv')
scaler = MinMaxScaler()

# Выполнение нормализации данных
normalized_data = scaler.fit_transform(data)

# Входные данные (возраст, давление в покое, максимальное давление)
X = data[['age', 'trestbps', 'thalach']].values

# Ожидаемые выходные данные (0 - низкий риск, 1 - высокий риск)
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)

# Создаем сложную нейронную сеть с тремя скрытыми слоями
model = Sequential()
model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель на данных
model.fit(X, y, epochs=1000, batch_size=16, validation_data=(X_test, y_test))

model.save('model.keras')
model = load_model('model.keras')
# Входные данные для предсказания
new_data = np.array([[59, 140, 164]])
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
# Делаем предсказание
prediction = model.predict(new_data)
print(prediction[0])
risk = 'высокий' if prediction[0] > 0.5 else 'низкий'

print(f"Результат предсказания: имеется {risk} риск сердечных заболеваний")