import copy
import random

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
import keras
import numpy

from classes.environments.snake import World
from classes.ml.genetic import Genetic

BOARD_SIZE = 8
NUM_POPULATION = 250


def generate_model():
    # Ref: https://www.youtube.com/watch?v=vhiO4WsHA6c (Chrispresso - AI Learns to play Snake!)
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(20, activation='sigmoid'))
    model.add(keras.layers.Dense(12, activation='sigmoid'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_model_input(world):
    # Ref: https://github.com/Chrispresso/SnakeAI/blob/master/snake.py at _vision_as_input_array()
    # There are 32 inputs to the neural networks:
    #        0 : Distance to wall
    #        1 : Distance to food
    #        2 : Distance to self
    #     3-23 : Same as above, but on different directions (there are 8 directions: E, SE, S, SW, W, NW, N, NE)
    #    24-27 : Current direction (one-hot encoding of E, S, W, N)
    #    28-31 : Tail direction (one-hot encoding of E, S, W, N)
    model_input = []
    for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
        x, y = world.snake.head_pos()
        total_dist = 0
        dist_to_food = None
        dist_to_self = None
        while True:
            x += dx
            y += dy
            total_dist += 1
            pos = (x, y)
            ch = world.get_ch(pos)
            if ch == World.CH_OBSTACLE:
                break
            elif ch == World.CH_FOOD:
                if dist_to_food is None:
                    dist_to_food = total_dist
            elif ch == World.CH_SNAKE:
                if dist_to_self is None:
                    dist_to_self = total_dist
        model_input.append(1 / total_dist)
        model_input.append(1 / dist_to_food if dist_to_food is not None else 0.0)
        model_input.append(1 / dist_to_self if dist_to_self is not None else 0.0)
    # One-hot encoding of snake head direction and snake tail direction
    for direction in world.snake.head_direction(), world.snake.tail_direction():
        if direction == World.DIRECTION_RIGHT:
            model_input += [1.0, 0.0, 0.0, 0.0]
        elif direction == World.DIRECTION_DOWN:
            model_input += [0.0, 1.0, 0.0, 0.0]
        elif direction == World.DIRECTION_LEFT:
            model_input += [0.0, 0.0, 1.0, 0.0]
        elif direction == World.DIRECTION_UP:
            model_input += [0.0, 0.0, 0.0, 1.0]
        else:
            model_input += [0.0, 0.0, 0.0, 0.0]
    assert len(model_input) == 32
    return model_input


def play_snake_game(model):
    world = World(board_size=BOARD_SIZE)
    world.init_snake()
    world.init_food()
    boards = []
    alive = True
    steps = 0
    idle_steps = 0
    while alive:
        board = copy.deepcopy(world.board)
        boards.append(board)

        # Feed forward
        model_input = generate_model_input(world)
        model_input_as_np = numpy.asarray([model_input])  # we only have one data, hence [model_input] and [0]
        direction_as_arr = model.predict(model_input_as_np)[0]
        direction_as_index = numpy.argmax(direction_as_arr)
        direction = World.ALL_DIRECTIONS[direction_as_index]

        ch = world.snake.move(direction)
        steps += 1
        idle_steps += 1
        if idle_steps > BOARD_SIZE * BOARD_SIZE:
            # We are circling
            break
        if ch == World.CH_EMPTY:
            pass
        elif ch == World.CH_OBSTACLE or ch == World.CH_SNAKE:
            alive = False
        elif ch == World.CH_FOOD:
            world.init_food()
            idle_steps = 0
        else:
            raise Exception(f'Invalid character encountered while playing snake game: {ch}')

    # We died
    snake_len = world.snake.len()
    # Ref: https://github.com/Chrispresso/SnakeAI/blob/master/snake.py at calculate_fitness()
    score = steps + ((2 ** snake_len) + (snake_len ** 2.1) * 500) - (((.25 * steps) ** 1.3) * (snake_len ** 1.2))
    return {
        'score': score,
        'snake_len': snake_len,
        'boards': boards
    }


def main():
    print('Generating individuals...')
    individuals = []
    for _ in range(NUM_POPULATION):
        individual = {
            'model': generate_model(),
            'score': None
        }
        individuals.append(individual)

    print('Evolution...')
    num_generation = 0
    while True:
        num_generation += 1
        print(f'Generation: {num_generation}')

        for individual in individuals:
            result = play_snake_game(individual['model'])
            individual['boards'] = result['boards']
            individual['score'] = result['score']

        # Print progress
        max_score = max([individual['score'] for individual in individuals])
        min_score = min([individual['score'] for individual in individuals])
        print(f'Score: {min_score:.1f} - {max_score:.1f}')

        # Survival of the fittest
        individuals = sorted(individuals, key=lambda individual: individual['score'], reverse=True)
        individuals = individuals[:NUM_POPULATION // 4]
        for board in individuals[0]['boards']:
            for y in board:
                print(''.join(y))
            print('-------')

        # Preserve parents as is
        new_individuals = []
        new_individuals += individuals

        # Now use the best individuals as template for the new generation
        while len(new_individuals) < NUM_POPULATION:
            # Roulette wheel selection
            parents = random.choices(population=individuals,
                                     weights=[individual['score'] for individual in individuals], k=2)
            parent1_weights = parents[0]['model'].get_weights()
            parent2_weights = parents[1]['model'].get_weights()
            child1_weights = []
            child2_weights = []
            # Crossover
            for n in range(len(parent1_weights)):
                child1_weights_layer, child2_weights_layer = Genetic.crossover_uniform(
                    parent1_weights[n], parent2_weights[n], prob_crossover=0.05
                )
                child1_weights.append(child1_weights_layer)
                child2_weights.append(child2_weights_layer)
            # Mutation
            for n in range(len(parent1_weights)):
                Genetic.mutate(child1_weights[n], prob_mutation=0.05, scale=0.2)
                Genetic.mutate(child2_weights[n], prob_mutation=0.05, scale=0.2)
            # Clip
            for n in range(len(parent1_weights)):
                child1_weights[n] = numpy.clip(child1_weights[n], -1, 1)
                child2_weights[n] = numpy.clip(child2_weights[n], -1, 1)
            # Construct
            child1_model = keras.models.clone_model(individuals[0]['model'])  # This does not copy the internal weights
            child1_model.set_weights(child1_weights)
            child2_model = keras.models.clone_model(individuals[0]['model'])
            child2_model.set_weights(child2_weights)
            new_individuals.append({
                'model': child1_model,
                'score': None
            })
            new_individuals.append({
                'model': child2_model,
                'score': None
            })
        individuals = new_individuals


if __name__ == '__main__':
    main()
