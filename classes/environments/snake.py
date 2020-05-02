import random


class Snake:
    def __init__(self, world):
        while True:
            pos = (random.randint(0, world.board_size - 1), random.randint(0, world.board_size - 1))
            if world.get_ch(pos) == World.CH_EMPTY:
                break
        world.put_ch(pos, World.CH_SNAKE_HEAD)
        self._world = world
        self._snake = [pos]

    def head_pos(self) -> (int, int):
        return self._snake[-1]

    def tail_pos(self) -> (int, int):
        return self._snake[0]

    def move(self, direction):
        if direction not in World.ALL_DIRECTIONS:
            raise Exception(f'Invalid direction: {direction}')
        (delta_x, delta_y) = direction
        (head_pos_x, head_pos_y) = self.head_pos()
        new_head_pos = (head_pos_x + delta_x, head_pos_y + delta_y)
        ch = self._world.get_ch(new_head_pos)
        if ch == World.CH_EMPTY:
            self._world.put_ch(self.head_pos(), World.CH_SNAKE)
            self._snake.append(new_head_pos)
            self._world.put_ch(self.head_pos(), World.CH_SNAKE_HEAD)
            self._world.put_ch(self.tail_pos(), World.CH_EMPTY)
            self._snake.pop(0)
            return ch
        elif ch == World.CH_OBSTACLE:
            return ch
        elif ch == World.CH_SNAKE:
            return ch
        elif ch == World.CH_FOOD:
            self._world.put_ch(self.head_pos(), World.CH_SNAKE)
            self._snake.append(new_head_pos)
            self._world.put_ch(self.head_pos(), World.CH_SNAKE_HEAD)
            return ch
        else:
            raise Exception(f'Invalid character encountered at pos {new_head_pos}: {ch}')


class Food:
    def __init__(self, world):
        while True:
            pos = (random.randint(0, world.board_size - 1), random.randint(0, world.board_size - 1))
            if world.get_ch(pos) == World.CH_EMPTY:
                break
        world.put_ch(pos, World.CH_FOOD)
        self._food_pos = pos

    def food_pos(self):
        return self._food_pos


class World:
    ALL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    CH_EMPTY = ' '
    CH_OBSTACLE = 'X'
    CH_SNAKE = '*'
    CH_SNAKE_HEAD = 'h'
    CH_FOOD = 'f'

    def __init__(self, board_size):
        # Empty board
        self.board_size = board_size
        self.board = [[World.CH_EMPTY for _ in range(board_size)] for _ in range(board_size)]
        # Borders
        for y in range(board_size):
            for x in range(board_size):
                if x == 0 or y == 0 or x == board_size - 1 or y == board_size - 1:
                    self.board[y][x] = World.CH_OBSTACLE
        # Critters
        self.snake = None
        self.food = None

    def init_snake(self):
        self.snake = Snake(self)

    def init_food(self):
        self.food = Food(self)

    def get_ch(self, pos) -> str:
        (x, y) = pos
        return self.board[y][x]

    def put_ch(self, pos, ch):
        (x, y) = pos
        self.board[y][x] = ch

    def draw(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                print(self.board[y][x], end='')
            print()
