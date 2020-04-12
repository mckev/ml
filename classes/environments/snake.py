import random


class Snake:

    def __init__(self, x, y):
        self.snake = [(x, y)]

    def head_pos(self) -> (int, int):
        return self.snake[len(self.snake) - 1]

    def tail_pos(self) -> (int, int):
        return self.snake[0]

    def move(self, board, direction):
        if direction not in Board.ALL_DIRECTIONS:
            raise Exception(f'Invalid direction: {direction}')
        (delta_x, delta_y) = direction
        (head_pos_x, head_pos_y) = self.head_pos()
        new_head_pos = (head_pos_x + delta_x, head_pos_y + delta_y)
        ch = board.get_ch(new_head_pos)
        if ch == Board.CH_EMPTY:
            board.put_ch(self.head_pos(), Board.CH_SNAKE)
            self.snake.append(new_head_pos)
            board.put_ch(self.head_pos(), Board.CH_SNAKE_HEAD)
            board.put_ch(self.tail_pos(), Board.CH_EMPTY)
            self.snake.pop(0)
            return ch
        elif ch == Board.CH_OBSTACLE:
            return ch
        elif ch == Board.CH_SNAKE:
            return ch
        elif ch == Board.CH_FOOD:
            board.put_ch(self.head_pos(), Board.CH_SNAKE)
            self.snake.append(new_head_pos)
            board.put_ch(self.head_pos(), Board.CH_SNAKE_HEAD)
            return ch
        else:
            raise Exception(f'Invalid character encountered at pos {new_head_pos}: {ch}')


class Board:
    ALL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    CH_EMPTY = ' '
    CH_OBSTACLE = 'X'
    CH_SNAKE = '*'
    CH_SNAKE_HEAD = 'h'
    CH_FOOD = 'f'

    def __init__(self, board_size):
        # Empty board
        self.board_size = board_size
        self.board = [[Board.CH_EMPTY for _ in range(board_size)] for _ in range(board_size)]
        # Draw obstacles
        for y in range(board_size):
            for x in range(board_size):
                if x == 0 or y == 0 or x == board_size - 1 or y == board_size - 1:
                    self.board[y][x] = Board.CH_OBSTACLE
        # Place snake randomly
        while True:
            self.snake = Snake(random.randint(0, board_size - 1), random.randint(0, board_size - 1))
            if self.get_ch(self.snake.head_pos()) == Board.CH_EMPTY:
                break
        self.put_ch(self.snake.head_pos(), Board.CH_SNAKE_HEAD)
        # Place food randomly
        self.place_food_randomly()

    def get_ch(self, pos) -> str:
        (x, y) = pos
        return self.board[y][x]

    def put_ch(self, pos, ch):
        (x, y) = pos
        self.board[y][x] = ch

    def place_food_randomly(self):
        while True:
            food_pos = (random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1))
            if self.get_ch(food_pos) == Board.CH_EMPTY:
                break
        self.put_ch(food_pos, Board.CH_FOOD)
        self.food_pos = food_pos

    def draw(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                print(self.board[y][x], end='')
            print()
