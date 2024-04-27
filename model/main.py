from csv import writer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = "data.csv"

CAPACITY = 100
K_P = 2
TIME = 100


class Queue:
    size: int
    capacity: int

    def __init__(self, capacity: int):
        self.size = 0
        self.capacity = capacity

    def add(self, n: int) -> int:
        """
        Tries to add `n` items to the queue. Returns the number of items that
        was added to the queue.
        """

        if self.size + n > self.capacity:
            added = self.capacity - self.size

            self.size = self.capacity

            return added

        self.size += n

        return n

    def remove(self, n: int) -> int:
        """
        Tries to remove `n` items from the queue. Returns the number of items
        that was removed from the queue.
        """

        if self.size - n < 0:
            removed = self.size

            self.size = 0

            return removed

        self.size -= n

        return n


class Controller:
    k_p: float

    def __init__(self, k_p: float):
        self.k_p = k_p

    def u(self, e: float) -> float:
        return self.k_p * e


def simulate():
    p = Queue(CAPACITY)
    c = Controller(K_P)

    with open(DATA_PATH, "w") as file:
        csv_writer = writer(file)

        csv_writer.writerow(["Time", "Desired", "Actual"])

        # A step function.
        r = np.concatenate([
            np.zeros(TIME // 2),
            (CAPACITY // 2) * np.ones(TIME // 2),
        ])

        for t in range(TIME):
            y = p.size

            csv_writer.writerow([t, r[t], y])

            e = r[t] - y

            u = c.u(e)
            u = int(u)
            u = max(0, u)

            p.add(u)

            consumed = np.random.normal(10, 2)
            consumed = int(consumed)

            assert consumed >= 0, "Consumed cannot be negative"

            p.remove(consumed)


def visualize():
    data = pd.read_csv(DATA_PATH)

    sns.lineplot(data=data, x="Time", y="Desired", label="Desired")
    sns.lineplot(data=data, x="Time", y="Actual", label="Actual")

    plt.show()


def main():
    simulate()
    visualize()


if __name__ == "__main__":
    main()
