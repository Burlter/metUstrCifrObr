def mother_wavelet_func(time):
    """returns vivlet function"""
    from numpy import exp

    return time * exp(- (time**2)/2)


def wavelet_plot(analisated_signal, wavelet_analisis_result):
    """Plot 2d and 3d graph of wavelet analisis result"""
    import matplotlib.pyplot as plt
    # import numpy as np

    fig = plt.figure()
    signal_ax = fig.add_subplot(211)
    signal_ax.plot(analisated_signal)
    signal_ax.set_title("Сигнал")
    two_d_wavelet_ax = fig.add_subplot(212)
    two_d_wavelet_ax.pcolormesh(wavelet_analisis_result)
    two_d_wavelet_ax.set_title("Двухмерное отображение вейвлет анализа")
    # fig_3d = plt.figure()
    # three_d_wavelet_ax = fig_3d.add_subplot(111, projection='3d')
    # x, y = np.meshgrid(np.arange(wavelet_analisis_result.shape[0]),
    #                    np.arange(wavelet_analisis_result.shape[1]))
    # three_d_wavelet_ax.set_title("Трёхмерное отображение вейвлет анализа")
    # three_d_wavelet_ax.plot_surface(x, y, wavelet_analisis_result,
    #                                 cmap='viridis', rcount=20, ccount=20)


def wavelet_analisis(wavelet_function, signal_to_analisis, wavelet_scale_arr,
                     wavelet_shift_arr):
    """Вейвлет анализ над переданным сигналом с помощью переданной вейвлет
    функции """
    import numpy as np

    wavelet_transform = []
    for column in range(0, len(wavelet_scale_arr)):
        wavelet_transform.append([])
        for row in range(0, len(wavelet_shift_arr)):
            wavelet_transform[column].append(
                sum(signal_to_analisis * wavelet_function(row, column)))
    wavelet_transform = np.array(wavelet_transform)

    wavelet_transform_intermediate = (wavelet_transform
                                      - np.min(wavelet_transform))
    wavelet_transform_normalizated = (wavelet_transform_intermediate
                                      / wavelet_transform_intermediate.max()
                                      * 255)

    return wavelet_transform_normalizated


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import pi, cos, sin
    from scipy.integrate import quad

    # Задаём общие характеристики сигналов
    signal_length = 3000
    signal_discretization = 1/1000
    signal_range = np.linspace(0, signal_length - 1, signal_length)
    signal_time = signal_range * signal_discretization

    # Формируем сигналы
    simple_signal_freq = 10
    simple_harmonic_signal = cos(simple_signal_freq * 2 * signal_time * pi/4)
    sum_of_harm_signals = (cos(7 * pi * signal_time)
                           + sin(2 * pi * signal_time))
    first_abrupt_freq = 3
    second_abrupt_freq = 10
    abrupt_change_signal = []
    for number in range(0, signal_length):
        if number > signal_length/3:
            abrupt_change_signal.append(2 * sin(2 * pi * signal_time[number]
                                        * first_abrupt_freq))
        else:
            abrupt_change_signal.append(cos(2 * pi * signal_time[number]
                                        * second_abrupt_freq))
    abrupt_change_signal = np.array(abrupt_change_signal)
    mother_wavelet = mother_wavelet_func(signal_time)

    # Добовляем помеху к сигналу
    white_noise = np.random.normal(0, 1, signal_length)
    simple_harmonic_signal_noised = simple_harmonic_signal + white_noise
    sum_of_harm_signals_noised = sum_of_harm_signals + white_noise
    abrupt_change_signal_noised = abrupt_change_signal + white_noise
    mother_wavelet_noised = mother_wavelet + white_noise

    signal_list = [simple_harmonic_signal, sum_of_harm_signals,
                   abrupt_change_signal, mother_wavelet,
                   simple_harmonic_signal_noised,
                   sum_of_harm_signals_noised, abrupt_change_signal_noised,
                   mother_wavelet_noised]

    # Проводим нормализацию для вейвлет функции
    wavelet_column = 100
    wavelet_row = 100
    wavelet_length = 1000
    normalization_check = quad(lambda x: mother_wavelet_func(x)**2, -np.inf,
                               np.inf)[0]
    if normalization_check < 0.99 or 1.01 < normalization_check:
        normalization_coeff = sum((mother_wavelet_func(
            np.linspace(0, wavelet_length, wavelet_length + 1)))**2)
        normalizated_mother_wavelet = (
            lambda time: ((mother_wavelet_func(time))
                          / np.sqrt(normalization_coeff)))
    else:
        normalizated_mother_wavelet = mother_wavelet_func()

    # Подбираем значения для диапазона изменения масштаба
    scale_min = 1/signal_length
    scale_max = 0.14
    scale_min_graph = (1/np.sqrt(scale_min)
                       * normalizated_mother_wavelet(
                           ((signal_range / signal_length) - 0.5)/scale_min))
    scale_max_graph = (1/np.sqrt(scale_max)
                       * normalizated_mother_wavelet(
                           ((signal_range / signal_length) - 0.5)/scale_max))
    wavelet_scale = (scale_min + (np.linspace(0, wavelet_column - 1,
                                              wavelet_column)
                                  / (wavelet_column - 1))
                     * (scale_max - scale_min))
    wavelet_shift = np.linspace(0, wavelet_row - 1, wavelet_row)/(wavelet_row
                                                                  - 1)

    # Рисуем гарфики минимума и макисимума диапазона
    plt.figure()
    plt.plot(scale_min_graph, label='График минимума "a"')
    plt.plot(scale_max_graph, label='График масимума "a"')
    plt.legend()

    # Формируем вейвлет функцию для работы
    wavelet = (
        lambda row_number, column_number: (
            (1/np.sqrt(wavelet_scale[column_number]))
            * normalizated_mother_wavelet(
                (signal_range / (signal_length - 1)
                 - wavelet_shift[row_number])
                / wavelet_scale[column_number])))

    # Вейвлет анализ всех сгенерированных нами сигналов
    for signal in signal_list:
        wavelet_transform_to_plot = wavelet_analisis(
            wavelet, signal, wavelet_scale, wavelet_shift)
        wavelet_plot(signal, wavelet_transform_to_plot)

    plt.show()


if __name__ == "__main__":
    main()
