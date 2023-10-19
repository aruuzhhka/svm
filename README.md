# svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Загрузка датасета о пассажирах Титаника
data = pd.read_csv('titanic.csv')
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Преобразование категориальных признаков в числовые
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Разделение данных на обучающую и тестовую выборки
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели SVM
svm_classifier = SVC()

# Обучение модели на обучающих данных
svm_classifier.fit(X_train, y_train)

# Предсказание классов на тестовых данных
y_pred = svm_classifier.predict(X_test)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM for Titanic Survival Classification: {accuracy}")
