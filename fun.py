def calculate_sum_odd_numbers_until(boundary_value):
    """
    Данная функция возвращает сумму всех нечётных чисел до заданного

    :boundary_value: Число до которого нужно произвести рассчёт
    :return: Сумма всех нечётных чисел
    """
    import numpy as np

    # Создаём массив счётов от 1 до заданного числа
    simple_array_to_point = np.linspace(1, boundary_value, boundary_value)

    # Возвращаем нечётные числа из массива и суммируем их
    return sum(simple_array_to_point[::2])

if __name__ == "__main__":
    sum_odd_numbers = calculate_sum_odd_numbers_until(8)
    print(sum_odd_numbers)



