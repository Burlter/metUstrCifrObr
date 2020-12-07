def plot_signal_difference(exponential_signal_delta, signal_stdev):
    """
    Функция выводит график разности между сигналами и среднеквадратическую
    ошибку фильтрации

    :exponential_signal_delta: Разность сигнала
    :signal_stdev: Среднеквадратическая ошибка сигнала
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(exponential_signal_delta)
    ax.set_title("Разность зашумлённого и фильтрованного сигналов")
    ax.text(0, 0.5, "Среднеквадратическая ошибка фильтрации равна " +
            "{}".format(round(signal_stdev, 3)), transform=ax.transAxes,
            fontsize=14)


def wiener_signal_filter(signal_length, signal_that_need_to_filter_length,
                         wiener_coefficients, noised_signal):
    """Signal filtration with wiener filter"""
    from numpy import array

    filtered_signal = []
    for number in range(0, signal_length):
        filtered_signal_step_sum = 0
        for sum_number in range(0, signal_that_need_to_filter_length):
            if number - sum_number >= 0:
                filtered_signal_step_sum += (
                    wiener_coefficients[sum_number]
                    * noised_signal[number - sum_number])
        filtered_signal.append(filtered_signal_step_sum)

    return array(filtered_signal)


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from statistics import stdev
    from numpy import exp, pi, cos

    # Set parameters for useful signal and find him
    useful_signal_length = 2**7
    signal_discretization = 1/useful_signal_length
    useful_signal_range = np.linspace(0, useful_signal_length - 1,
                                      useful_signal_length)
    # useful_signal = (
    #     cos(pi * useful_signal_range * signal_discretization /
    #         useful_signal_length)
    #     - cos(3 * pi * useful_signal_range * signal_discretization /
    #           useful_signal_length))
    useful_signal = (cos(2 * pi * useful_signal_range * 3
                     * signal_discretization))**3

    # Plot useful signal
    plt.figure()
    plt.plot(useful_signal)
    plt.title("Полезный сигнал")
    plt.xlabel("Отсчёты")
    plt.ylabel("Амплитуда")

    # Set parametrs for main signal and generate him
    main_signal_length = 1500
    shift_size = 500
    main_signal = []
    for number in range(0, main_signal_length):
        if shift_size < number and number < shift_size + useful_signal_length:
            main_signal.append(useful_signal[number - shift_size])
        else:
            main_signal.append(0)
    main_signal = np.array(main_signal)

    # Generate gaussian noise
    noise_dispersion = 0.9
    gaussian_noise = np.random.normal(0, noise_dispersion, main_signal_length)
    gaussian_noise_signal = main_signal + gaussian_noise

    # Generate exponential correlation function noise, which formula
    # has been taken from "Быков В.В. Цифровое моделирование в статистич
    # радиотехнике" page 104, № 1
    distribution_of_a_random_variable = 1
    exponential_operator = (
        distribution_of_a_random_variable * signal_discretization)
    noise_exp = exp(-exponential_operator)
    gaussian_noise_for_exp = np.random.normal(0, 1, main_signal_length)
    exponentional_noise = [noise_dispersion * np.sqrt(1 - noise_exp**2) *
                           gaussian_noise_for_exp[number]]
    for number in range(1, main_signal_length):
        exponentional_noise.append(
                exponentional_noise[number - 1] * noise_exp
                + gaussian_noise_for_exp[number] * noise_dispersion * np.sqrt(
                    1 - noise_exp**2))
    exponentional_noise = np.array(exponentional_noise)

    exponentional_noise_signal = main_signal + exponentional_noise

    # Find correlation exponentional noise and theoretical correlation
    exponentional_noise_correlation = np.correlate(
        exponentional_noise, exponentional_noise, mode="full")\
        / exponentional_noise.size

    correlation_shift_range = np.linspace(0, main_signal_length,
                                          main_signal_length)
    theoretical_exponentional_noise_correlation = (
        noise_dispersion**2 * np.exp(-distribution_of_a_random_variable
                                     * correlation_shift_range
                                     * signal_discretization))

    # Plot theoretical and practical correlation
    plt.figure()
    plt.title("График сравнения атокорреляциий")
    plt.xlabel("Отсчёты")
    plt.ylabel("Амплитуда")
    plt.plot(exponentional_noise_correlation[
        int(exponentional_noise_correlation.size//2):], label="Исходная")
    plt.plot(theoretical_exponentional_noise_correlation,
             label="Теоритическая")
    plt.legend()

    # Find coeficients for gaussian noise signal filter
    usful_signal_correlation = []
    expected_value = sum(useful_signal) / useful_signal_length
    for number in useful_signal_range:
        usful_signal_correlation_step_sum = 0
        for sum_number in range(0, useful_signal_length - int(number)):
            usful_signal_correlation_step_sum += (((useful_signal[sum_number]
                                                    - expected_value)
                                                   * (useful_signal[
                                                       sum_number
                                                       + int(number)]
                                                      - expected_value)))
        usful_signal_correlation.append(usful_signal_correlation_step_sum
                                        / (useful_signal_length))
    usful_signal_correlation = np.array(usful_signal_correlation)

    autocorrelation_useful_signal_matrix = []
    for i in useful_signal_range:
        autocorrelation_useful_signal_matrix.append([])
        for j in useful_signal_range:
            autocorrelation_useful_signal_matrix[int(i)].append(
                usful_signal_correlation[abs(int(i) - int(j))])
    autocorrelation_useful_signal_matrix = np.array(
        autocorrelation_useful_signal_matrix)

    autocorrelation_gaussian_noise_matrix = []
    for i in useful_signal_range:
        autocorrelation_gaussian_noise_matrix.append([])
        for j in useful_signal_range:
            if i == j:
                autocorrelation_gaussian_noise_matrix[int(i)].append(
                    noise_dispersion**2)
            else:
                autocorrelation_gaussian_noise_matrix[int(i)].append(0)
    autocorrelation_gaussian_noise_matrix = np.array(
        autocorrelation_gaussian_noise_matrix)

    cross_correlation_matrix = (autocorrelation_gaussian_noise_matrix
                                + autocorrelation_useful_signal_matrix)

    gaussian_noise_wiener_coefficients = (np.matmul(np.linalg.inv(
        cross_correlation_matrix), usful_signal_correlation))

    gaussian_noise_filtered_signal = wiener_signal_filter(
        main_signal_length, useful_signal_length,
        gaussian_noise_wiener_coefficients, gaussian_noise_signal)

    plt.figure()
    plt.plot(gaussian_noise_signal, '--', label="Зашумлённый сигнал")
    plt.plot(main_signal, label="Сигнал без шума")
    plt.plot(gaussian_noise_filtered_signal, label="Фильтрованный сигнал",
             linewidth=3)
    plt.legend()

    gaussian_signal_delta = main_signal - gaussian_noise_filtered_signal
    gaussian_signal_stdev = (stdev(gaussian_signal_delta)
                             / stdev(main_signal))
    plot_signal_difference(gaussian_signal_delta, gaussian_signal_stdev)

    # Find coeficients for filter signal with exponential noise
    autocorrelation_exponential_noise_matrix = []
    for i in useful_signal_range:
        autocorrelation_exponential_noise_matrix.append([])
        for j in useful_signal_range:
            if i == j:
                autocorrelation_exponential_noise_matrix[int(i)].append(
                    noise_dispersion**2 * noise_exp)
            else:
                autocorrelation_exponential_noise_matrix[int(i)].append(0)
    autocorrelation_exponential_noise_matrix = np.array(
        autocorrelation_exponential_noise_matrix)

    cross_correlation_matrix = (autocorrelation_exponential_noise_matrix
                                + autocorrelation_useful_signal_matrix)

    exponential_noise_wiener_coefficients = (np.matmul(np.linalg.inv(
        cross_correlation_matrix), usful_signal_correlation))

    exponentional_noise_filtered_signal = wiener_signal_filter(
        main_signal_length, useful_signal_length,
        exponential_noise_wiener_coefficients, exponentional_noise_signal)

    plt.figure()
    plt.plot(exponentional_noise_signal, '--', label="Зашумлённый сигнал")
    plt.plot(main_signal, label="Сигнал без шума")
    plt.plot(exponentional_noise_filtered_signal, label="Фильтрованный сигнал",
             linewidth=2)
    plt.legend()

    exponential_signal_delta = main_signal - exponentional_noise_signal
    exponential_signal_stdev = (stdev(exponential_signal_delta)
                                / stdev(main_signal))
    plot_signal_difference(exponential_signal_delta, exponential_signal_stdev)

    plt.show()


if __name__ == "__main__":
    main()
