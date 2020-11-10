# Const
N = 3000 

def plot_signal_difference(signal_delta, signal_stdev):
    """
    Функция выводит график разности между сигналами и среднеквадратическую ошибку фильтрации

    :signal_delta: Разность сигнала
    :signal_stdev: Среднеквадратическая ошибка сигнала
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(signal_delta)
    ax.set_title("Разность зашумлённого и аппроксимированного сигналов")
    ax.text(0, 0.5, "Среднеквадратическая ошибка аппроксимации равна {}".format(round(signal_stdev, 3)), transform=ax.transAxes, fontsize=15)


def signal_generate(time, function):
    """ 
    Генерирует сигнал с заданными параметрами

    :time: Отсчёты сигнала
    :function: Функция по которой строиться график
    :returns: Сигнал с заданными параметрами
    """
    from numpy import log, pi, cos 

    generated_signal = eval(function)

    return generated_signal 


def matrix_multiply(first_matrix, second_matrix):
    """
    Перемножает матрицы

    :first_matrix: первая матрица
    :second_matrix: вторая матрица
    :return: Результат перемножения
    """
    import numpy as np

    multiply_result = np.zeros((len(first_matrix), len(second_matrix[0])))
    for row in range(0, len(first_matrix)):
        for column in range(0, len(second_matrix[0])):
            for counter in range(0, len(second_matrix)):
                multiply_result[row][column] += first_matrix[row][counter] * second_matrix[counter][column]

    return multiply_result


def approximate_signal_with_basis(approximate_order, signal_for_smooth, non_orthogonal_basis):
    """
    Сглаживает сигнал неортогональным базисом определённого порядка

    :approximate_order: Порядок базиса
    :signal_for_smooth: Сигнал для сглаживания
    :non_orthogonal_basis: Неортогональный базис
    :return: фильтрованный сигнал
    """
    import numpy as np

    coefficient_of_decomposition = []
    for order_number in range(0, approximate_order):
        coefficient_of_decomposition.append(sum(signal_for_smooth * non_orthogonal_basis[order_number])/ \
                                            sum(non_orthogonal_basis[order_number] * non_orthogonal_basis[order_number]))

    # Осуществляем аппроксимацию на основе базисной функции
    approximated_signal = []
    for number in range(0, N):
        approximated_signal.append(sum(coefficient_of_decomposition * non_orthogonal_basis[:approximate_order,number]))

    return np.array(approximated_signal) 


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import legendre
    from pprint import pprint
    from statistics import stdev

    # Генерируем сигнал
    time = np.linspace(0, 2, N)
    signal_function = 'log(2 - cos(2 * pi * 3 * time))'
    generated_signal = signal_generate(time, signal_function)
    
    # Генерируем шумы
    gaussian_noise_signal = generated_signal + np.random.normal(0, 1, N)
    
    # Генерируем полином Лежандра
    M = int(N/30)
    numbers_range = np.linspace(-1, 1, N)
    generated_legendre = []
    for order in range(1, M):
        legendre_polynomial = legendre(order - 1)
        generated_legendre.append(legendre_polynomial(numbers_range))
    generated_legendre = np.array(generated_legendre)

    # Выводим на график несколько базисных функций
    plt.figure()
    for order_number in range(0, 6):
        plt.plot(numbers_range, generated_legendre[order_number][:], label="P{}(x)".format(order_number))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Pn(x)")
    plt.title("Полином лежандра")
    
    # Проверяем условие ортоганальности
    transponated_generated_legendre = np.transpose(generated_legendre[:22])
    orthogonality_check = matrix_multiply(generated_legendre[:22], transponated_generated_legendre)
    pprint(orthogonality_check)

    # Находим зависимость среднеквадратической ошибки от порядка базиса и вычисляем индекс минимальной
    # среднеквадратической ошибки
    stdev_array = []
    plot_start = 10
    orders_range = range(plot_start, len(generated_legendre))
    for order_number in orders_range:
        approximated_signal = approximate_signal_with_basis(order_number, gaussian_noise_signal, generated_legendre)
        signal_delta = generated_signal - approximated_signal
        signal_stdev = stdev(signal_delta)/stdev(generated_signal)
        stdev_array.append(signal_stdev)
    index_of_minimum_stdev = stdev_array.index(min(stdev_array)) + plot_start

    # Рисуем график зависимости
    plt.figure()
    plt.xlabel("Порядки базиса")
    plt.ylabel("Величина среднеквадратической ошибки")
    plt.plot(orders_range, stdev_array)
    plt.text(plot_start, min(stdev_array), "Наименьшее значение среднеквадратической ошибки = {}".format(round(min(stdev_array), 3)), fontsize=14)
    plt.text(plot_start, min(stdev_array) + 0.1, "Порядок наименьшего значения среднеквадратической ошибки = {}".format(index_of_minimum_stdev), fontsize=14)

    # Т. к. матрица не ортогональна, то коэф. параметрической модели сигнала будем искать соответственно
    approximated_signal = approximate_signal_with_basis(index_of_minimum_stdev, gaussian_noise_signal, generated_legendre)
    signal_delta = generated_signal - approximated_signal
    signal_stdev = stdev(signal_delta)/stdev(generated_signal)

    # Рисуем графики сигналов
    plt.figure()
    plt.title("Сравнение зашумлённого сигнала и фильтрованного аппроксимацией")
    plt.xlabel("Время")
    plt.ylabel("Амплитуда")
    plt.plot(time, gaussian_noise_signal, label="Зашумлённый сигнал")
    plt.plot(time, approximated_signal, label="Аппроксимированный сигнал")
    plt.legend()

    # Выводим график разности и находим среднеквадратическую ошибку
    plot_signal_difference(signal_delta, signal_stdev)

    plt.show()


if __name__ == "__main__":
    main()
