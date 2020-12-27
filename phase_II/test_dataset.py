import matplotlib.pyplot as plt
import numpy as np


def test_binary_file(index: int) -> None:
    data_path = "dataSet/train/data.bin"
    label_path = "dataSet/train/labels.bin"
    image_from_file = np.memmap(str(data_path), dtype=np.uint8, offset=index * (81 * 81 * 3), mode='r')
    image_from_file = image_from_file[:(81 * 81 * 3)].reshape(81, 81, 3)
    plt.imshow(image_from_file)
    crop_size = [81, 81, 3]

    with open(f'{data_path}', "r") as data_set:
        with open(f'{label_path}', "r") as labels:
            data_set = np.memmap(str(data_path), dtype='uint8', mode='r', shape=(81, 81, 3),
                                 offset=crop_size[0] * crop_size[1] * crop_size[2] * index)
            label = np.memmap(str(label_path), dtype='uint8', mode='r', shape=(1,), offset=index)
            plt.imshow(data_set)
            plt.title("Traffic light" if label else "Not Traffic light")
            plt.show()


if __name__ == '__main__':
    test_binary_file(2)
