# Контрольная работа по математическому моделированию
### Владимир Кирилкин ПИ18-2, вариант 31
## Задание 1
Система имеем 12 дискретных состояний. Изменение состояний происходит в дискретные моменты времени с заданной вероятность. Схема марковского процесса изображена на рисунке.
![](.github/given_chain.png)
Требуется определить:
1) Вероятность того, что за 9 шагов система перейдет из состояния 7 в состояние 5
2) Вероятности состояний системы спустя 6 шагов, если в начальный момент вероятность состояний были следующими `A = (0,01; 0,11; 0,09; 0; 0,06; 0,12; 0,08; 0,12; 0,12; 0,07; 0,14; 0,08)`
3) Вероятность первого перехода за 9 шагов из состояния 10 в состояние 8
4) Вероятность перехода из состояния 5 в состояние 2 не позднее чем за 9 шагов
5) Среднее количество шагов для перехода из состояния 1 в состояние 3
6) Вероятность первого возвращения в состояние 7 за 10 шагов
7) Вероятность возвращения в состояние 3 не позднее чем за 10 шагов
8) Среднее время возвращения в состояние 1
9) Установившиеся вероятности
#### Входные данные
|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | 0 | 0 | 0 | 0.25 | 0.04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2** | 0.14 | 0 | 0.24 | 0.24 | 0 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 0 | 0.25 | 0 | 0 | 0 | 0.46 | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 0.13 | 0.04 | 0.21 | 0 | 0.08 | 0 | 0.12 | 0.11 | 0.05 | 0 | 0 | 0 |
| **5** | 0 | 0.43 | 0 | 0 | 0 | 0 | 0.06 | 0.13 | 0.24 | 0 | 0 | 0 |
| **6** | 0 | 0.13 | 0.07 | 0 | 0.41 | 0 | 0 | 0 | 0.13 | 0.21 | 0 | 0 |
| **7** | 0 | 0 | 0 | 0.1 | 0.09 | 0 | 0 | 0 | 0 | 0 | 0.2 | 0.35 |
| **8** | 0 | 0 | 0 | 0 | 0.19 | 0.11 | 0.14 | 0 | 0.17 | 0 | 0.16 | 0 |
| **9** | 0 | 0 | 0 | 0.35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.59 |
| **10** | 0 | 0 | 0 | 0.48 | 0.39 | 0.07 | 0 | 0 | 0 | 0 | 0 | 0 |
| **11** | 0 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0 | 0.49 | 0 | 0 | 0.13 |
| **12** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.26 | 0.01 | 0.19 | 0 | 0 |
#### Код
```python
import csv
from functools import lru_cache

import numpy as np
from numpy.linalg import matrix_power


class MarkovChain:
    matrix: np.array

    def __init__(self, matrix: np.array):
        self.matrix = matrix

    def __str__(self):
        return str(self.matrix)

    @classmethod
    def from_file(cls, filename: str) -> "MarkovChain":
        matrix = []
        with open(filename, encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                matrix.append(list(map(float, row)))
        return cls(np.array(matrix))

    def reach(self, of: int, to: int, by: int) -> float:
        """ Вероятность перехода в состояние """
        return matrix_power(self.matrix, by)[of, to]

    def condition_chances(self, start_chances: np.array, by: int) -> np.array:
        """ Вероятности состояний системы через `by` шагов с данными начальными вероятностями """
        return np.dot(matrix_power(self.matrix, by), start_chances)

    def make_step(self, matrix: np.array) -> np.array:
        new = np.zeros(self.matrix.shape)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                s = 0
                for m in range(self.matrix.shape[0]):
                    s += self.matrix[i, m] * matrix[m, j] if m != j else 0
                new[i, j] = s
        return new

    def first_reach(self, of: int, to: int, by: int) -> float:
        """ Вероятность первого перехода """
        final = self.matrix
        for step in range(2, by + 1):
            final = self.make_step(final)
        return final[of, to]

    def no_longer_reach(self, of: int, to: int, by: int) -> float:
        """ Вероятность перехода не позднее чем """
        return sum(self.first_reach(of, to, step) for step in range(1, by + 1))

    def mean_reach(self, of: int, to: int) -> float:
        """ Среднее количество шагов для перехода"""
        temp_matrix = self.matrix
        res = temp_matrix[of, to]
        for step in range(2, 1000):
            temp_matrix = self.make_step(temp_matrix)
            res += step * temp_matrix[of, to]
        return res

    @lru_cache()
    def first_return(self, of: int, by: int) -> float:
        """ Вероятность первого возвращения """
        s = np.zeros(self.matrix.shape)
        for step in range(1, by):
            s += self.first_return(of, step) * matrix_power(self.matrix, by - step)
        return (matrix_power(self.matrix, by) - s)[of, of]

    def no_longer_return(self, of: int, by: int) -> float:
        """ Вероятность возвращения не позднее чем """
        return sum(self.first_return(of, step) for step in range(1, by + 1))

    def mean_return(self, of: int) -> float:
        """ Среднее время возвращения """
        return sum(step * self.first_return(of, step) for step in range(1, 130))

    def stable_condition(self) -> np.array:
        """ Установивишиеся вероятности """
        m = self.matrix.T - np.eye(self.matrix.shape[0])
        m[-1, :] = 1
        b = np.array([0] * (self.matrix.shape[0] - 1) + [1])
        x = np.dot(np.linalg.inv(m), b)
        return x


if __name__ == "__main__":
    chain = MarkovChain.from_file("matrix.csv")

    task1 = chain.reach(7 - 1, 5 - 1, 9)
    print(f"Вероятность того, что за 9 шагов система перейдет из состояния 7 в состояние 5: {task1: .3f}")

    task2 = chain.condition_chances(
        np.array((0.01, 0.11, 0.09, 0, 0.06, 0.12, 0.08, 0.12, 0.12, 0.07, 0.14, 0.08)), 6,
    )
    print("Вероятности состояний системы спустя 6 шагов с заданными начальными вероятностями: ")
    print(task2)

    task3 = chain.first_reach(10 - 1, 8 - 1, 9)
    print(f"Вероятность первого перехода за 9 шагов из состояния 10 в состояние 8: {task3: .3f}")

    task4 = chain.no_longer_reach(5 - 1, 2 - 1, 9)
    print(f"Вероятность перехода из состояния 5 в состояние 2 не позднее чем за 9 шагов: {task4: .3f}")

    task5 = chain.mean_reach(1 - 1, 3 - 1)
    print(f"Среднее количество шагов для перехода из состояния 1 в состояние 3: {task5: .3f}")

    task6 = chain.first_return(7 - 1, 10)
    print(f"Вероятность первого возвращения в состояние 7 за 10 шагов: {task6: .3f}")

    task7 = chain.no_longer_return(3 - 1, 10)
    print(f"Вероятность возвращения в состояние 3 не позднее чем за 10 шагов {task7: .3f}")

    task8 = chain.mean_return(1 - 1)
    print(f"Среднее время возвращения в состояние 1 {task8: .3f}")

    task9 = chain.stable_condition()
    print("Установившиеся вероятности: ")
    print(task9)
```
#### Ответы
![](.github/screen_1.png)
## Задание 2
Задана система массового обслуживания со следующими характеристиками:
- интенсивность поступления			`λ=33`
- каналов обслуживания				`m=2`
- интенсивность обслуживания		`μ=24`
- максимальный размер очереди		`n=16`

Изначально требований в системе нет
1)	Составьте граф марковского процесса, запишите систему уравнений Колмогорова и найдите установившиеся вероятности состояний.
2)	Найдите вероятность отказа в обслуживании.
3)	Найдите относительную и абсолютную интенсивность обслуживания.
4)	Найдите среднюю длину в очереди.
5)	Найдите среднее время в очереди.
6)	Найдите среднее число занятых каналов.
7)	Найдите вероятность того, что поступающая заявка не будет ждать в очереди.
8)	Найти среднее время простоя системы массового обслуживания.
9)	Найти среднее время, когда в системе нет очереди.
#### Код
```python
import numpy as np


class SMO:
    matrix: np.array
    stable_condition: np.array

    @staticmethod
    def create_matrix(arriving: int, channels: int, intensity: int, max_size: int) -> np.array:
        """ Создание матрицы переходов """
        matrix = np.zeros((channels + max_size + 1, channels + max_size + 1))
        for i in range(channels + max_size):
            matrix[i, i + 1] = arriving
            matrix[i + 1, i] = intensity * (i + 1) if i < channels else intensity * channels
        return matrix

    @staticmethod
    def create_stable_condition(matrix: np.array) -> np.array:
        """ Установивишиеся вероятности """
        diagonal = np.diag([matrix[i, :].sum() for i in range(matrix.shape[0])])
        new = matrix.T - diagonal
        new[-1, :] = 1
        zeros = np.zeros(new.shape[0])
        zeros[-1] = 1
        return np.linalg.inv(new).dot(zeros)

    def __init__(self, arriving: int, channels: int, intensity: int, max_size: int):
        self.arriving = arriving
        self.channels = channels
        self.intensity = intensity
        self.max_size = max_size
        self.matrix = self.create_matrix(arriving, channels, intensity, max_size)
        self.stable_condition = self.create_stable_condition(self.matrix)

    def rejection(self) -> float:
        """ Вероятность отказа в обслуживании """
        return self.stable_condition[-1]

    def bandwidth(self, absolute=False) -> float:
        """ Пропускная способность """
        bandwidth = 1 - self.stable_condition[-1]
        return bandwidth * self.arriving if absolute else bandwidth

    def mean_length(self) -> float:
        """ Средняя длина очереди """
        return sum(i * self.stable_condition[self.channels + i] for i in range(1, self.max_size + 1))

    def mean_time(self) -> float:
        """ Среднее время в очереди """
        return sum(
            (i + 1) / (self.channels * self.intensity) * self.stable_condition[self.channels + i]
            for i in range(self.max_size)
        )

    def mean_busy_channels(self) -> float:
        """ Среднее число занятых каналов """
        return sum(i * self.stable_condition[i] for i in range(1, self.channels + 1)) + sum(
            self.channels * self.stable_condition[i]
            for i in range(self.channels + 1, self.channels + self.max_size + 1)
        )

    def wont_wait(self) -> float:
        """ Вероятность что поступающая заявка не будет ждать в очереди """
        return sum(self.stable_condition[: self.channels])

    def mean_stand(self) -> float:
        """ Среднее время простоя """
        return 1 / self.arriving

    def mean_no_queue(self) -> float:
        """ Среднее время, когда в системе нет очереди """
        probabilities = {i: self.stable_condition[i] for i in range(self.channels + 1)}
        normal_coefficient = 1 / sum(probabilities.values())
        probabilities = {k: v * normal_coefficient for k, v in probabilities.items()}
        return self.time_in_subset(set(probabilities.keys()), probabilities, 0.0005, 10)

    def time_in_subset(self, subset: set, probabilities: dict, time_step: float, time: int) -> float:
        """
        Время пребывания в подмножестве
        :param subset: список состояний
        :param probabilities: вероятности
        :param time_step: шаг времени
        :param time: полное время проведения испытаний
        :return: время нахождения
        """
        k_coefficient = np.zeros(self.matrix.shape)
        for i in subset:
            for j in range(self.matrix.shape[0]):
                k_coefficient[j][i] = self.matrix[i][j]
        for i in subset:
            for j in range(self.matrix.shape[0]):
                if j != i:
                    k_coefficient[i][i] -= self.matrix[i][j]

        not_subset = set(range(self.matrix.shape[0])) - subset
        f = 0
        for t in np.arange(time_step, time + time_step, time_step):
            probabilities = self.find_p(k_coefficient, subset, probabilities, time_step)
            f_transition = sum(probabilities[i] * self.matrix[i][j] for j in not_subset for i in subset)
            f += t * f_transition
        return f * time_step

    @staticmethod
    def find_p(k_coefficient, subset, probabilities, time) -> list:
        k = {x: [0, 0, 0, 0] for x in subset}
        for x in subset:
            for i in subset:
                k[x][0] += k_coefficient[x][i] * probabilities[i]
            for i in subset:
                k[x][1] += (k_coefficient[x][i]) * (probabilities[i] + k[x][0] * time / 2)
            for i in subset:
                k[x][2] += (k_coefficient[x][i]) * (probabilities[i] + k[x][1] * time / 2)
            for i in subset:
                k[x][3] += (k_coefficient[x][i]) * (probabilities[i] + k[x][2] * time)
        return [(probabilities[x] + time * (k[x][0] + 2 * k[x][1] + 2 * k[x][2] + k[x][3]) / 6) for x in subset]


if __name__ == "__main__":
    smo = SMO(33, 2, 24, 16)
    print("Установившиеся вероятности:")
    print(smo.stable_condition)
    print(f"Вероятность отказа в обслуживании: {smo.rejection(): .7f}")
    print(f"Относительная интенсивность обслуживания: {smo.bandwidth(): .7f}")
    print(f"Абсолютная интенсивность обслуживания: {smo.bandwidth(absolute=True): .7f}")
    print(f"Средняя длина очереди: {smo.mean_length(): .7f}")
    print(f"Среднее время в очереди: {smo.mean_time(): .7f}")
    print(f"Среднее число занятых каналов: {smo.mean_busy_channels(): .7f}")
    print(f"Вероятность того, что поступающая заявка не будет ждать в очереди: {smo.wont_wait(): .7f}")
    print(f"Среднее время простоя СМО: {smo.mean_stand(): .7f}")
    print(f"Среднее время, когда в системе нет очереди: {smo.mean_no_queue(): .7f}")
    print("Матрица интенсивностей:")
    print(*map(list, smo.matrix), sep="\n")
```
#### Ответы
![](.github/screen_2.png)