import random


class World:
    DIRECTION_LEFT = (-1, 0)
    DIRECTION_RIGHT = (1, 0)
    DIRECTION_UP = (0, -1)
    DIRECTION_DOWN = (0, 1)
    ALL_DIRECTIONS = [DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_UP, DIRECTION_DOWN]

    CH_EMPTY = ' '
    CH_OBSTACLE = 'X'
    CH_SNAKE = '*'
    CH_SNAKE_HEAD = 'h'
    CH_FOOD = 'f'

    def __init__(self, board_size: int):
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

    def init_snake(self, pos=None) -> None:
        self.snake = Snake(self, pos)

    def init_food(self, pos=None) -> None:
        self.food = Food(self, pos)

    def get_ch(self, pos) -> str:
        (x, y) = pos
        return self.board[y][x]

    def put_ch(self, pos, ch) -> None:
        (x, y) = pos
        self.board[y][x] = ch

    def draw(self) -> None:
        for y in range(self.board_size):
            for x in range(self.board_size):
                print(self.board[y][x], end='')
            print()


class Snake:
    def __init__(self, world: World, pos=None):
        if pos is None:
            while True:
                pos = (random.randint(0, world.board_size - 1), random.randint(0, world.board_size - 1))
                if world.get_ch(pos) == World.CH_EMPTY:
                    break
        world.put_ch(pos, World.CH_SNAKE_HEAD)
        self._world: World = world
        self._snake_pos = [pos]

    def head_pos(self) -> (int, int):
        return self._snake_pos[-1]

    def tail_pos(self) -> (int, int):
        return self._snake_pos[0]

    def move(self, direction) -> str:
        if direction not in World.ALL_DIRECTIONS:
            raise Exception(f'Invalid direction: {direction}')
        (delta_x, delta_y) = direction
        (head_pos_x, head_pos_y) = self.head_pos()
        new_head_pos = (head_pos_x + delta_x, head_pos_y + delta_y)
        ch = self._world.get_ch(new_head_pos)
        if ch == World.CH_EMPTY:
            self._world.put_ch(self.head_pos(), World.CH_SNAKE)
            self._snake_pos.append(new_head_pos)
            self._world.put_ch(self.head_pos(), World.CH_SNAKE_HEAD)
            self._world.put_ch(self.tail_pos(), World.CH_EMPTY)
            self._snake_pos.pop(0)
            return ch
        elif ch == World.CH_OBSTACLE:
            return ch
        elif ch == World.CH_SNAKE:
            return ch
        elif ch == World.CH_FOOD:
            self._world.put_ch(self.head_pos(), World.CH_SNAKE)
            self._snake_pos.append(new_head_pos)
            self._world.put_ch(self.head_pos(), World.CH_SNAKE_HEAD)
            return ch
        else:
            raise Exception(f'Invalid character encountered at pos {new_head_pos}: {ch}')

    def get_visions(self, vision_len: int):
        snake_head_x, snake_head_y = self.head_pos()
        visions = []
        for y in range(snake_head_y - vision_len, snake_head_y + vision_len + 1):
            vision = []
            for x in range(snake_head_x - vision_len, snake_head_x + vision_len + 1):
                if 0 <= x < self._world.board_size and 0 <= y < self._world.board_size:
                    ch = self._world.get_ch((x, y))
                else:
                    ch = World.CH_OBSTACLE
                vision.append(ch)
            visions.append(vision)
        return visions


class Food:
    def __init__(self, world: World, pos=None):
        if pos is None:
            while True:
                pos = (random.randint(0, world.board_size - 1), random.randint(0, world.board_size - 1))
                if world.get_ch(pos) == World.CH_EMPTY:
                    break
        world.put_ch(pos, World.CH_FOOD)
        self._food_pos = pos

    def food_pos(self) -> (int, int):
        return self._food_pos
