import unittest

import numpy

from classes.ml.genetic import Genetic


class TestGenetic(unittest.TestCase):
    matrix1 = numpy.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])

    matrix2 = numpy.array([
        [101.0, 102.0, 103.0],
        [104.0, 105.0, 106.0],
        [107.0, 108.0, 109.0],
        [110.0, 111.0, 112.0],
        [113.0, 114.0, 115.0]
    ])

    def test_crossover_uniform(self):
        child1, child2 = Genetic.crossover_uniform(TestGenetic.matrix1, TestGenetic.matrix2, prob_crossover=0.3)
        total_rows, total_cols = TestGenetic.matrix1.shape
        num_crossover = 0
        for y in range(total_rows):
            for x in range(total_cols):
                if child1[y][x] == TestGenetic.matrix1[y][x]:
                    self.assertTrue(child2[y][x] == TestGenetic.matrix2[y][x])
                else:
                    num_crossover += 1
                    self.assertTrue(child1[y][x] == TestGenetic.matrix2[y][x])
                    self.assertTrue(child2[y][x] == TestGenetic.matrix1[y][x])
        percent_crossover = num_crossover / (total_rows * total_cols)
        self.assertTrue(0.1 < percent_crossover < 0.5)  # Not always true (this is only for documentation)

    def test_crossover_single_point(self):
        child1, child2 = Genetic.crossover_single_point(TestGenetic.matrix1, TestGenetic.matrix2)
        total_rows, _ = TestGenetic.matrix1.shape
        crossover_row = None  # row where it crossover
        for y in range(total_rows):
            if crossover_row is None and child1[y].tolist() == TestGenetic.matrix1[y].tolist():
                crossover_row = y
            if crossover_row is None:
                self.assertTrue(child1[y].tolist() == TestGenetic.matrix2[y].tolist())
                self.assertTrue(child2[y].tolist() == TestGenetic.matrix1[y].tolist())
            else:
                self.assertTrue(child1[y].tolist() == TestGenetic.matrix1[y].tolist())
                self.assertTrue(child2[y].tolist() == TestGenetic.matrix2[y].tolist())
        self.assertTrue(0 <= crossover_row < total_rows)

    def test_crossover_multi_points(self):
        child1, child2 = Genetic.crossover_multi_points(TestGenetic.matrix1, TestGenetic.matrix2, prob_crossover=0.3)
        total_rows, _ = TestGenetic.matrix1.shape
        num_crossover = 0
        for y in range(total_rows):
            if child1[y].tolist() == TestGenetic.matrix1[y].tolist():
                self.assertTrue(child2[y].tolist() == TestGenetic.matrix2[y].tolist())
            else:
                num_crossover += 1
                self.assertTrue(child1[y].tolist() == TestGenetic.matrix2[y].tolist())
                self.assertTrue(child2[y].tolist() == TestGenetic.matrix1[y].tolist())
        percent_crossover = num_crossover / total_rows
        self.assertTrue(0.1 < percent_crossover < 0.5)  # Not always true (this is only for documentation)

    def test_mutate(self):
        matrix1 = numpy.copy(TestGenetic.matrix1)
        Genetic.mutate(matrix1, prob_mutation=0.3, scale=0.2)
        total_rows, total_cols = matrix1.shape
        total_mutation = 0
        for y in range(total_rows):
            for x in range(total_cols):
                if matrix1[y][x] != TestGenetic.matrix1[y][x]:
                    total_mutation += 1
                    delta = abs(TestGenetic.matrix1[y][x] - matrix1[y][x])
                    self.assertLess(delta, 1.0)
        percent_mutation = total_mutation / (total_rows * total_cols)
        self.assertTrue(0.1 < percent_mutation < 0.5)  # Not always true (this is only for documentation)
