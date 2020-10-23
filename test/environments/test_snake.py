import unittest

from classes.environments.snake import World


class TestSnakeWorld(unittest.TestCase):

    def test_init_world(self):
        for _ in range(100):
            world = World(board_size=4)
            self.assertEqual(world.board_size, 4)
            self.assertEqual(world.board, [
                ['X', 'X', 'X', 'X'],
                ['X', ' ', ' ', 'X'],
                ['X', ' ', ' ', 'X'],
                ['X', 'X', 'X', 'X']
            ])

            self.assertIsNone(world.snake)
            world.init_snake()
            snake_head_x, snake_head_y = world.snake.head_pos()
            self.assertTrue(1 <= snake_head_x <= 2)
            self.assertTrue(1 <= snake_head_y <= 2)

            self.assertIsNone(world.food)
            world.init_food()
            food_pos_x, food_pos_y = world.food.food_pos()
            self.assertTrue(1 <= food_pos_x <= 2)
            self.assertTrue(1 <= food_pos_y <= 2)

            self.assertTrue(world.snake.head_pos() != world.food.food_pos())

    def test_snake_movements(self):
        world = World(board_size=4)
        world.init_snake((2, 1))
        world.init_food((1, 2))
        self.assertEqual(world.board, [
            ['X', 'X', 'X', 'X'],
            ['X', ' ', 'h', 'X'],
            ['X', 'f', ' ', 'X'],
            ['X', 'X', 'X', 'X']
        ])
        ch = world.snake.move(World.DIRECTION_DOWN)
        self.assertEqual(ch, ' ')
        self.assertEqual(world.board, [
            ['X', 'X', 'X', 'X'],
            ['X', ' ', ' ', 'X'],
            ['X', 'f', 'h', 'X'],
            ['X', 'X', 'X', 'X']
        ])
        ch = world.snake.move(World.DIRECTION_LEFT)
        self.assertEqual(ch, 'f')
        self.assertEqual(world.board, [
            ['X', 'X', 'X', 'X'],
            ['X', ' ', ' ', 'X'],
            ['X', 'h', '*', 'X'],
            ['X', 'X', 'X', 'X']
        ])
        ch = world.snake.move(World.DIRECTION_UP)
        self.assertEqual(ch, ' ')
        self.assertEqual(world.board, [
            ['X', 'X', 'X', 'X'],
            ['X', 'h', ' ', 'X'],
            ['X', '*', ' ', 'X'],
            ['X', 'X', 'X', 'X']
        ])
        ch = world.snake.move(World.DIRECTION_UP)
        self.assertEqual(ch, 'X')
        self.assertEqual(world.board, [
            ['X', 'X', 'X', 'X'],
            ['X', 'h', ' ', 'X'],
            ['X', '*', ' ', 'X'],
            ['X', 'X', 'X', 'X']
        ])
