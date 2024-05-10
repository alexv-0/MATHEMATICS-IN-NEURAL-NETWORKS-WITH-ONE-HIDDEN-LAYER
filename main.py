import numpy
# scipy.special expit() - сигмоида
import scipy.special
# scipy.ndimage for для поворота изображений
import scipy.ndimage

import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, nodes : list[int, 3], learningrate : int):
        # Колличество слоев
        self.n = 3
        
        # Список колличеств нейронов на входном, вложенных и выходном слоях
        self.nodes = nodes
        
        # Создание матриц весов W и V
        self.W = numpy.random.normal(0.0, pow(self.nodes[0], -0.5), (self.nodes[0], self.nodes[1]))
        self.V = numpy.random.normal(0.0, pow(self.nodes[1], -0.5), (self.nodes[1], self.nodes[2]))

        # Темп обучения
        self.lr = learningrate
        
        # Функция активации - сигмоида
        self.act_func = lambda x: scipy.special.expit(x)


    # Тренировка
    def train(self, inputs_list : list, targets_list : list) -> None:
        # Перевод списка входных и списка целевых данных в матрицы-строки
        # и их транспонирование в матрицы-столбцы
        I = numpy.array(inputs_list,  ndmin=2).T
        To = numpy.array(targets_list, ndmin=2).T
        
        # Вычисление матриц-столбцов скрытого и выходного слоев
        H = self.act_func(numpy.dot(self.W.T, I))
        O = self.act_func(numpy.dot(self.V.T, H))

        # Метод градиентного спуска
        Ri = I; Rh = H; Ro = O;
        self.V += self.lr * numpy.transpose((To - Ro) * Ro * (1 - Ro) * Rh.T)
        # Ошибка на скрытом слое считается пропорционально весам
        Th = numpy.dot(self.V, (To-Ro) )
        self.W += self.lr * numpy.transpose((Th - Rh) * Rh * (1 - Rh) * Ri.T)
    
    # Запрос в НС
    def query(self, inputs_list : list):
        # Перевод списка входных данных в матрицу-строку
        # и ее транспонирование в матрицу-стобец
        I = numpy.array(inputs_list,  ndmin=2).T
        
        # Вычисление матриц-столбцов скрытого и выходного слоев
        H = self.act_func(numpy.dot(self.W.T, I))
        O = self.act_func(numpy.dot(self.V.T, H))
        
        return O

# Формат данных: {digit, pix1_1, pix1_2, pix1_3, ... pix28_28}, где:
#     digit - нарисованное  число от 0 до 9
#     pixA_B - яркость от 0 до 255 (чб) пикселя на А-ой строке и в В-ом столбце
def open_training_data() -> list[str]:
    # Перевод тренировочных даных MNIST из CSV формата в список
    with open("mnist_dataset/mnist_train.csv", 'r') as training_data_file:
        training_data_list = training_data_file.readlines()
    return training_data_list

def open_test_data() -> list[str]:
    # Перевод тестовых даных MNIST из CSV Формата в список
    with open("mnist_dataset/mnist_test.csv", 'r') as test_data_file:
        test_data_list = test_data_file.readlines()
    return test_data_list

def train(NN : NeuralNetwork, training_data_list : list[str], epochs : int) -> None:
    # Для каждой эпохи
    for _ in range(epochs):
        # Пройти по всем записям в тренировочных данных
        for record in training_data_list:

            # Выделение чисел из строки, разделенных запятой
            values = record.split(',')
            # Перевод данных из диапазона [0, 255] в [0.01, 1]
            inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

            # Создание целевых выходных данных 
            # (все по 0.01, по правильному индексу 0.99)
            targets = numpy.zeros(NN.nodes[-1]) + 0.01
            targets[int(values[0])] = 0.99

            NN.train(inputs, targets)

def test(NN : NeuralNetwork, test_data_list : list[str]) -> list[bool]:
    # Тестирование НС

    # Список для оценки качества работы НС, изначально пустой
    scorecard = []

    # Пройтись по всем записям тестовых данных
    for record in test_data_list:
        # Выделяем числа из записи по запятой
        values = record.split(',')
        # Ответ - это первое значение в списке:
        correct_digit = int(values[0])
        # перевод данных из диапазона [0, 255] в [0.01, 1]
        inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        
        # Запрос в НС
        outputs = NN.query(inputs)
        # Индекс максимального значения - это ответ НС
        felt_digit = numpy.argmax(outputs)
        # Фиксирование результатов в список
        scorecard.append(felt_digit == correct_digit)
    
    return scorecard

def makeTest(nodes : list[int, 3], learning_rate : int, epochs : int) -> float:
    # Эпохи (epochs) - количество тренировок НС с одним набором данных
    # Узлы (nodes) - количество нейронов в слоях
    # Темп обычения (learningRate) - коэффициент при градиентном спуске

    # Инициализация экземпляра НС
    NN = NeuralNetwork(nodes, learning_rate)
    
    # Тренировка
    print(f"Тренировка нейронной сети: NN(nodes={nodes}, lr={learning_rate}) при epoches={epochs} ...... ", end='')
    training_data_list = open_training_data()
    train(NN=NN,
          training_data_list=training_data_list,
          epochs=epochs
    )
    
    # Тестирование
    test_data_list = open_test_data()
    scorecard = test(
        NN=NN,
        test_data_list=test_data_list,
    )
    res = sum(scorecard) / len(scorecard)
    print(f"Эффективность составляет {res}")

    return res

def plot(test1, test2):
    # mock data
    #test1 = {"0.01": [(1, 50), (2, 60), (3, 75), (5, 55), (7, 10)],
    #         "0.02": [(1, 55), (2, 63), (4, 76), (5, 67), (7, 20)]}
    lrs = [0.01, 0.02]

    # Создание графиков
    fig1, ax1 = plt.subplots()
    for lr in lrs:
        ax1.plot([xy[0] for xy in test1[str(lr)]],
                 [xy[1] for xy in test1[str(lr)]],
                 '-o',
                 label=f"lr={lr}")
        #for (x, y) in test1[str(lr)]:
        #    ax1.text(x+0.1, y+0.1, f"{y}")
    ax1.legend()
    ax1.set_xlabel('Колличество эпох')
    ax1.set_ylabel('Эффективность нейронной сети')
    ax1.set_title('График зависимости эфективности нейронной сети от количества эпох')

    #mock data
    #test2 = [(10, 70), (50, 75), (100, 85)]

    fig2, ax2 = plt.subplots()
    ax2.plot([xy[0] for xy in test2],
             [xy[1] for xy in test2],
             '-o',
             label="lr=0.01\nepochs=5")
    #for (x, y) in test2:
    #    ax2.text(x+0.1, y+0.1, f"{y}")
    ax2.legend()
    ax2.set_xlabel('Колличество нейронов на скрытом слое')
    ax2.set_ylabel('Эффективность нейронной сети')
    ax2.set_title('График зависимости эфективности нейронной сети от количества нейронов на скрытом слое')

def main():
    # Тесты 1 - на оптимальный темп обучения
    # и оптимальное количество эпох
    test1 = {"0.01": [],
             "0.02": []}
    epochs = [1] + [x*3 for x in range(1, 21)]
    lrs = [0.01, 0.02]
    for lr in lrs:
        for epoch in epochs:
            res = makeTest(nodes=[784, 200, 10],
                    learning_rate=lr,
                    epochs=epoch,
            )
            test1[str(lr)].append((epoch, res))
    print(test1)

    # Тесты 2 - нахождение зависимости эффективности
    # нейронной сети от колличества нейронов на скрытом слое
    test2 = []
    hiddens = [5, 10, 20, 50, 100, 200, 300, 500, 1000]
    for hidden in hiddens:
        res = makeTest(nodes=[784, hidden, 10],
                learning_rate=0.01,
                epochs=15,
        )
        test2.append((hidden, res))
    print(test2)

    # Создание графиков
    plot(test1, test2)

if (__name__=="__main__"):
    main()