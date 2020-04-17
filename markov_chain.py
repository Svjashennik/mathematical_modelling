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
        s = 0
        for step in range(1, by + 1):
            s += self.first_reach(of, to, step)
        return s

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
        s = 0
        for step in range(1, by + 1):
            s += self.first_return(of, step)
        return s

    def mean_return(self, of: int) -> float:
        """ Среднее время возвращения """
        s = 0
        for step in range(1, 130):
            s += step * self.first_return(of, step)
        return s

    def stable_condition(self) -> np.array:
        m = self.matrix.T - np.eye(self.matrix.shape[0])
        m[-1, :] = 1
        b = np.array([0] * (self.matrix.shape[0] - 1) + [1])
        x = np.dot(np.linalg.inv(m), b)
        return x


if __name__ == "__main__1":
    chain = MarkovChain.from_file("matrix.csv")
    print(chain.mean_reach(0, 2))

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
