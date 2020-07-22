from sklearn.neural_network import MLPRegressor  # импорт нейросети
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import train_test_split  # функция для разделения выборки на обучающую и тестовую

import pandas as pd  # импорт pandas
import numpy as np

import os

import matplotlib.pyplot as plt

FEATURE_COLS = ['people', 'people_1', 'people_2', 'families', 'death']


# загрузка данных из excel
def get_df_data():
    assert os.path.isfile('test.xlsx'), 'Ошибка! Файла не существует'
    excel_data_df = pd.read_excel('test.xlsx', sheet_name='Отчет', skiprows=3, header=None,
                                  names=['year', 'B', 'people', 'families', 'death'])
    assert not excel_data_df.empty, 'Ошибка! Данные в файле отсутствуют'

    excel_data_df.drop('B', axis=1, inplace=True)  # убираем лишнюю колонку
    excel_data_df['year'] = excel_data_df['year'].str.split(' ').str.get(0).astype(
        int)  # превращаем строку  "1990 г." в целое число 1990
    excel_data_df['people_1'] = excel_data_df['people'].shift(
        -1)  # получаем целевую переменную путем сдвига значений на 1 вверх
    excel_data_df['people_2'] = excel_data_df['people'].shift(
        -2)  # получаем целевую переменную путем сдвига значений на 1 вверх
    excel_data_df['predict'] = excel_data_df['people'].shift(-3)

    excel_data_df['families'].fillna(excel_data_df['families'].mean(), inplace=True)
    excel_data_df['death'].fillna(excel_data_df['death'].mean(), inplace=True)

    return excel_data_df


def extract_X_y(df):
    X_for_predict = df.loc[
        df.year == 2017, FEATURE_COLS].values.reshape(1,
                                                      -1)  # отбираем 1 образец на 2019 год чтобы предсказать для 2020
    df.dropna(axis=0, inplace=True)  # убираем строки, где есть незаполненные признаки
    X = df[FEATURE_COLS]  # формируем выборку признаков
    y = df['predict']  # формируем целевую переменную
    return X, y, X_for_predict


def MLP(X, y):
    assert not (X.empty and y.empty), 'Ошибка! Х и у не могут быть пустыми'
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=1)  # разделение выборки на обучающую и тестовую
    regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(5, 7, 4)).fit(X_train,
                                                                                        y_train)  # обучение нейросети
    print(f'MLP score:{regr.score(X_test, y_test)}')  # Точность предсказания
    return regr


def grnn(X, y):
    assert not (X.empty and y.empty), 'Ошибка! Х и у не могут быть пустыми'
    from neupy import algorithms

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    nw = algorithms.GRNN(std=1000, verbose=False)  # алгоритм зависит только от параметров и тренировочной выборки
    nw.train(x_train, y_train)
    y_predicted = nw.predict(x_test).flatten()
    err = 1 - np.sum((y_predicted - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)  # R2 error
    print(f'GRNN score:{err}')
    return nw


def linearSVR(X,y):
    assert not (X.empty and y.empty), 'Ошибка! Х и у не могут быть пустыми'
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=1)  # разделение выборки на обучающую и тестовую
    regr = LinearSVR().fit(X_train,y_train)  # обучение алгоритрма
    print(f'linearSVR score:{regr.score(X_test, y_test)}')  # Точность предсказания
    return regr



if __name__ == '__main__':
    df = get_df_data()
    X, y, X_for_predict = extract_X_y(df.copy())

    plt.plot(df['year'], df['people'], 'b', label='people')

    df = df.append(pd.DataFrame({'year': [2020]}))  # добавляет строку с 2020 в датафрейм
    predictors = [('grnn', grnn(X, y), 'g'),
                  ('MLP', MLP(X, y), 'r'),
                  ('linearSVR',linearSVR(X,y),'y')]

    for name, alg, colour in predictors:  # цикл который рисует график
        df[name] = np.append(alg.predict(df[FEATURE_COLS].dropna().values),
                             [np.nan, ] * 3)  # предсказание
        df[name] = df[name].shift(3)  # сдвиг предсказания на 3
        plt.plot(df['year'], df[name], colour, label=name)  # нарисовать график предсказания
        print(f'prediction:{alg.predict(X_for_predict)}')
    plt.legend()
    plt.show()
