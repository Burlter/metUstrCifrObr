def plot_signal_difference(signal_delta, signal_stdev):
    """
    Функция выводит график разности между сигналами и среднеквадратическую
    ошибку фильтрации

    :signal_delta: Разность сигнала
    :signal_stdev: Среднеквадратическая ошибка сигнала
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(signal_delta)
    ax.set_title("Разность зашумлённого и аппроксимированного сигналов")
    ax.text(0, 0.5, "Среднеквадратическая ошибка аппроксимации равна " +
            "{}".format(round(signal_stdev, 3)), transform=ax.transAxes,
            fontsize=14)


def approximate_signal_with_basis(window_length, signal_for_smooth, basis):
    """
    Сглаживает сигнал методом наименьших квадратов с помощью скользящего окна

    :window_length: Длинна скользящего окна
    :signal_for_smooth: Сигнал для сглаживания
    :basis: базис
    :return: фильтрованный сигнал
    """
    import numpy as np

    approximated_signal = []
    for number in range(0, len(signal_for_smooth)):

        # Двигаем окно
        if number < int(window_length/2):
            cutted_signal = np.zeros(window_length)

            for signal_number in range(int(window_length/2) - number,
                                       window_length):
                cutted_signal[signal_number] = signal_for_smooth[
                        signal_number - int(window_length/2) + number
                        ]

        elif number > (len(signal_for_smooth) - int(window_length/2)
                       - (window_length % 2)):
            cutted_signal = np.zeros(window_length)
            for signal_number in range(0, int(window_length/2)
                                       + (len(signal_for_smooth) - number)):
                cutted_signal[signal_number] = signal_for_smooth[
                        signal_number + number - int(window_length/2)]
        else:
            cutted_signal = signal_for_smooth[
                    number - int(window_length/2):
                    number + int(window_length/2) + window_length % 2]
        cutted_signal = np.array(cutted_signal)

        # Находим коэффициент разложения
        transpose_basis = basis.transpose()
        coefficient_of_decomposition = np.matmul(np.linalg.inv(np.matmul(
            transpose_basis, basis)),
            np.matmul(transpose_basis, cutted_signal))

        # Осуществляем аппроксимацию на основе базисной функции
        approximated_signal.append(sum(
            np.multiply(coefficient_of_decomposition,
                        basis[int(window_length/2)
                              + window_length % 2])))

    return np.array(approximated_signal)


def least_squares_polinomial_generate(window_length, basis_order_range):
    """Генерирует мнк полином с заданной длинной окна и порядком базиса"""
    import numpy as np

    moving_window_range = np.linspace(0, window_length - 1, window_length)

    least_squares_array = []
    for order in basis_order_range:
        least_squares_array.append(
                (moving_window_range/np.sqrt(window_length))**order)

    return np.array(least_squares_array).transpose()


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from statistics import stdev
    from numpy import log, pi, cos

    # Генерируем сигнал
    main_signal_length = 3000
    time = np.linspace(0, 2, main_signal_length)
    generated_signal = log(2 - cos(2 * pi * 3 * time))

    # Генерируем шумы
    gaussian_noise_signal = (generated_signal
                             + np.random.normal(0, 1, main_signal_length))

    # Задаём константные значения для базисной функции
    order_max = 3
    order_range = np.linspace(0, order_max - 1, order_max)
    moving_window_max = 380

    # Находим зависимость среднеквадратической ошибки от длинны окна и
    # вычисляем индекс минимальной среднеквадратической ошибки.
    stdev_array = []
    plot_start = 280
    window_range = range(plot_start, moving_window_max)
    for window_number in window_range:
        # Вычисляем полином для конкретного размера окна
        polinominal = least_squares_polinomial_generate(window_number,
                                                        order_range)

        # Аппроксимируем сигнал полученным полиномом
        approximated_signal = approximate_signal_with_basis(
                window_number, gaussian_noise_signal, polinominal)

        # Вычисляем среднеквадратическую ошибку
        signal_delta = generated_signal - approximated_signal
        signal_stdev = stdev(signal_delta)/stdev(generated_signal)
        stdev_array.append(signal_stdev)

    index_of_minimum_stdev = stdev_array.index(min(stdev_array)) + plot_start

    # Рисуем график зависимости
    plt.figure()
    plt.xlabel("Порядки базиса")
    plt.ylabel("Величина среднеквадратической ошибки")
    plt.plot(window_range, stdev_array)
    plt.text(plot_start, min(stdev_array), "Наименьшее значение " +
             "среднеквадратической ошибки = " +
             "{}".format(round(min(stdev_array), 3)), fontsize=12)
    plt.text(plot_start, min(stdev_array) + 0.001, "Порядок наименьшего" +
             "значения среднеквадратической ошибки = " +
             "{}".format(index_of_minimum_stdev), fontsize=12)

    # Аппроксимируем наилучшей длинной окна, которую мы узнали ранее
    # Снова формируем полином
    polinominal = least_squares_polinomial_generate(index_of_minimum_stdev,
                                                    order_range)

    # Аппроксимируем
    approximated_signal = approximate_signal_with_basis(index_of_minimum_stdev,
                                                        gaussian_noise_signal,
                                                        polinominal)
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

    # Выводим график разности сигналов
    plot_signal_difference(signal_delta, signal_stdev)

    plt.show()


if __name__ == "__main__":
    main()
