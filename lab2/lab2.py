# Const
N = 18000

def plot_signal_difference(main_signal, main_like_signal, counts):
    """
    Функция выводит график разности между сигналами и среднеквадратическую ошибку фильтрации

    :main_signal: Главный сигнал
    :main_like_signal: Сигнал, который хочет быть похожим на главный
    :counts: Отсчёты
    """
    import matplotlib.pyplot as plt
    from statistics import stdev

    signal_delta = main_signal - main_like_signal
    epsilon_spline = stdev(signal_delta)/stdev(main_signal)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(signal_delta)
    ax.text(0, 0.5, "Среднеквадратическая ошибка интерполяции равна {}".format(epsilon_spline), transform=ax.transAxes, fontsize=15)

def get_points(input_signal, points_number, step):
    """
    Получить узловые точки из сигнала с заданным шагом

    :input_signal: Сигнал для которого нужно выделить узловые точки
    :points_number: Количество точек
    :step: Шаг с которым будут браться точки
    :return: Массив узловых точек
    """
    import numpy as np

    calculated_points = []
    if len(input_signal) % step != 0:
        for segment in range(0, points_number - 1):
            calculated_points.append(np.median(input_signal[segment * step:(segment + 1) * step]))
        calculated_points.append(np.median(input_signal[(points_number - 1) * step:]))
    else:
        for segment in range(0, points_number):
            calculated_points.append(np.median(input_signal[segment * step:(segment + 1) * step]))

    return np.array(calculated_points)


def tuke_noise_generate(epsilon, dispersion1, dispersion2, mean):
    """
    Генерирует шум по модели Тюке

    :epsilon: Вероятность зашумления
    :dispersion1: Дисперсия несущего сигнала
    :dispersion2: Дисперсия зашумления
    :mean: Мат. ожидание
    :return: Массив значений по модели Тюке
    """
    import numpy as np

    first_noise_signal = np.random.normal(mean, dispersion1, N)
    second_noise_signal = np.random.normal(mean, dispersion2, N)

    noise_probability = np.random.uniform(0, 1, N)
    
    noise = []
    for number in range(0, N):
        if noise_probability[number] <= (1 - epsilon):
            noise.append(first_noise_signal[number])
        else:
            noise.append(second_noise_signal[number])

    return np.array(noise)


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


def interpolation_signal_plot(high_discretization_signal, high_discretization_numbers, low_discretization_signal,
                              low_discretization_numbers, interpolated_signal, noise_type):
    """
    Строит три графика с заданной дискретизацией

    :high_discretization_signal: Сигнал с высокой частотой дискретизации
    :high_discretization_numbers: Отсчёты сигнала с высокой частотой дискретизации
    :low_discretization_signal: Сигнал с низкой частотой дискретизации
    :low_discretization_numbers: Отсчёты сигнала с низкой частотой дискретизации
    :input_signal: Сигнал с низкой частотой дискретизации прошедший интерполяцию
    :noise_type: Тип шума
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    plt.title("Интерполяция {} шума".format(noise_type))
    plt.xlabel('Отсчёты')
    plt.ylabel('Амплитуда')
    plt.plot(high_discretization_numbers, high_discretization_signal, label="Исходный сигнал")
    plt.plot(low_discretization_numbers, low_discretization_signal, "o", label="Узловые точки исходного сигнала")
    plt.plot(high_discretization_numbers, interpolated_signal, "--",color="r", label="Интерполянт")
    plt.legend()


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    # Генерируем сигнал
    time = np.linspace(0, 2, N)
    signal_function = 'log(2 - cos(2 * pi * 3 * time))'
    generated_signal = signal_generate(time, signal_function)

    # Генерируем шумы
    gaussian_noise_signal = generated_signal + np.random.normal(0, 1, N)
    tuke_noise_signal = generated_signal + tuke_noise_generate(0.05, 0.25, 4.5, 0)

    # Генерируем массив узловых точек из медианных значений окон с шагом 
    M = 120
    points_time = np.linspace(0, 2, M)
    window_length = int(N/M)
    points_of_tuke_noise_signal = get_points(tuke_noise_signal, M, window_length)
    points_of_gaussian_noise_signal = get_points(gaussian_noise_signal, M, window_length)

    # Кубическая интерполяция для сигнала с гауссовским шумом
    gaussian_noise_signal_interpolation = CubicSpline(points_time, points_of_gaussian_noise_signal)
    interpolation_signal_plot(gaussian_noise_signal, time, points_of_gaussian_noise_signal, points_time, gaussian_noise_signal_interpolation(time), "Гауссовского")
    plot_signal_difference(generated_signal, gaussian_noise_signal_interpolation(time), time)

    # Кубическая интерполяция для сигнала с шумом Тюке
    tuke_noise_signal_interpolation = CubicSpline(points_time, points_of_tuke_noise_signal)
    interpolation_signal_plot(tuke_noise_signal, time, points_of_tuke_noise_signal, points_time, tuke_noise_signal_interpolation(time), "Тюке")
    plot_signal_difference(generated_signal, tuke_noise_signal_interpolation(time), time)

    plt.show()


if __name__ == "__main__":
    main()
