import os
import numpy as np
from PIL import Image
from phase_I.run_attention import find_tfl_lights
from phase_II.data_augmentation import darken_image, bright_image, add_noise


def write_label_to_binary_file(dir_name: str, label: int) -> None:
    with open(f"dataSet/{dir_name}/labels.bin", "ab") as file_name:
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))


def write_image_to_binary_file(dir_name: str, image: Image, label: int) -> None:
    with open(f"dataSet/{dir_name}/data.bin", "wb") as file_name:
        image.tofile(file_name, sep="", format="%s")
        write_label_to_binary_file(dir_name, label)


def crop_and_save_img(dir_name: str, image: Image, x_coordinate: int, y_coordinate: int, label: int) -> None:
    cropped_img = image.crop((x_coordinate - 41, y_coordinate - 41, x_coordinate + 40, y_coordinate + 40))
    write_image_to_binary_file(dir_name, np.array(cropped_img).astype(np.uint8), label)

    if dir_name == 'train':
        write_image_to_binary_file(dir_name, np.fliplr(np.array(cropped_img)).astype(np.uint8), label)
        write_image_to_binary_file(dir_name, darken_image(cropped_img).astype(np.uint8), label)
        write_image_to_binary_file(dir_name, bright_image(cropped_img).astype(np.uint8), label)
        write_image_to_binary_file(dir_name, add_noise(cropped_img).astype(np.uint8), label)


def crop_image_by_coordinates(dir_name: str, origin_img: Image, coordinates: np.array, label: int) -> None:
    x_coordinates, y_coordinates = coordinates[0], coordinates[1]
    x = x_coordinates[int(len(x_coordinates) / 2)]
    y = y_coordinates[int(len(x_coordinates) / 2)]
    crop_and_save_img(dir_name, origin_img, x, y, label)


def find_not_tfl(image: np.array, tfl_coordinates: np.ndarray) -> np.array:
    tfl_suspicious = find_tfl_lights(image)
    tfl_suspicious = (tfl_suspicious[0] + tfl_suspicious[2], tfl_suspicious[1] + tfl_suspicious[3])
    not_tfl = ([], [])

    for x, y in zip(tfl_coordinates[0], tfl_coordinates[1]):
        for i, j in zip(tfl_suspicious[0], tfl_suspicious[1]):

            if x != i or y != j and all([m != 19 for m in image[x - 41: x + 41][y - 41: y + 41]]):
                not_tfl[0].append(i)
                not_tfl[1].append(j)

    return not_tfl


def get_images_list(dir_name: str) -> list:
    path = f'data/labelIds/{dir_name}/'
    img_list = [file for file in os.listdir(path)]

    return img_list


def prepare_data_set(dir_name: str) -> None:
    img_list = get_images_list(dir_name)

    for img_name in img_list:
        image = Image.open(f"data/labelIds/{dir_name}/{img_name}")
        tfl_coordinates = np.where(np.array(image) == 19)

        if tfl_coordinates[0].any():
            origin_img = Image.open(f"data/leftImg8bit/{img_name[:-20]}_leftImg8bit.png")

            crop_image_by_coordinates(dir_name, origin_img, tfl_coordinates, 1)
            crop_image_by_coordinates(dir_name, origin_img, find_not_tfl(np.array(origin_img), tfl_coordinates), 0)


def set_data() -> None:
    prepare_data_set("train")
    prepare_data_set("val")


if __name__ == '__main__':
    set_data()
