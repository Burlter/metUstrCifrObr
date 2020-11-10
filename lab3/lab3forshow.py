# Const
N = 1800

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


def polynomial_legendre_generate(basis_functions_number):
    """
    Генерирует полином Лежандра с заданным количиством базовых функций

    :basis_functions_number: Количество базовых функций
    :return: Полином Лежандра
    """
    from scipy.special import legendre
    import numpy as np

    numbers_range = np.linspace(-1, 1, N)
    legendre_polynomial = legendre(basis_functions_number - 1)

    return legendre_polynomial(numbers_range)


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


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from pprint import pprint

    # Генерируем сигнал
    time = np.linspace(0, 2, N)
    signal_function = 'log(2 - cos(2 * pi * 3 * time))'
    generated_signal = signal_generate(time, signal_function)
    
    # Генерируем шумы
    gaussian_noise_signal = generated_signal + np.random.normal(0, 1, N)
    
    # Генерируем полином Лежандра
    M = int(N/100)
    generated_legendre = []
    for order in range(1, M + 1):
        generated_legendre.append(polynomial_legendre_generate(order))

    # Проверяем условие ортоганальности
    transponated_generated_legendre = np.transpose(generated_legendre)
    orthogonality_check = matrix_multiply(generated_legendre, transponated_generated_legendre)
    pprint(orthogonality_check)

    plt.show()


if __name__ == "__main__":
    main()
