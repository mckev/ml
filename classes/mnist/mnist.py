from typing import List


class Mnist:
    IMAGE_SIZE = 28

    @staticmethod
    def show_mnist_data(mnist_data):
        pos = 0
        stdout = [f'Number: {mnist_data["number"]}']
        for _ in range(mnist_data['image_size']):
            line = ''
            for _ in range(mnist_data['image_size']):
                b = mnist_data['image_bytes'][pos]
                if b > 127:
                    line += '*'
                else:
                    line += ' '
                pos += 1
            stdout.append(line)
        return stdout

    @staticmethod
    def retrieve_mnist_datas(filename: str) -> List[any]:
        mnist_datas = []
        print(f'Reading MNIST file {filename}')
        with open(filename, 'rb') as f:
            while True:
                number_in_byte = f.read(1)
                if not number_in_byte:
                    break
                image_bytes = f.read(Mnist.IMAGE_SIZE * Mnist.IMAGE_SIZE)
                mnist_data = {
                    'number': int.from_bytes(number_in_byte, byteorder='little'),
                    'image_size': Mnist.IMAGE_SIZE,
                    'image_bytes': image_bytes
                }
                mnist_datas.append(mnist_data)
        print(f'Total {len(mnist_datas)} records')
        return mnist_datas
