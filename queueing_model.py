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
