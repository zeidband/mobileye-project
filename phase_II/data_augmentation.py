import imgaug.augmenters as iaa


def darken_image(image):
    contrast = iaa.GammaContrast(gamma=2.0)
    contrast_image = contrast.augment_image(image)

    return contrast_image


def bright_image(image):
    contrast = iaa.GammaContrast(gamma=0.5)
    contrast_image = contrast.augment_image(image)

    return contrast_image


def add_noise(image):
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    noise_image = gaussian_noise.augment_image(image)

    return noise_image

