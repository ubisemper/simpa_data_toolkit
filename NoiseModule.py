import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from scipy.ndimage.filters import convolve

"""
Functions to generate motion blur artifacts
"""

def create_empty_kernel(kernel_size):
    return np.zeros((kernel_size, kernel_size), dtype=int)

def horizontal_blur(image, kernel_size, factor):
    kernel = create_empty_kernel(kernel_size)
    kernel[:, kernel_size // 2] = 1
    kernel = factor * kernel
    return convolve(image, kernel)


def vertical_blur(image, kernel_size, factor):
    kernel = create_empty_kernel(kernel_size)
    kernel[kernel_size // 2, :] = 1
    kernel = factor * kernel
    return convolve(image, kernel)


def diagonal_blur(image, kernel_size, factor):
    kernel = factor * np.eye(kernel_size)
    return convolve(image, kernel)


def random_blur(image, kernel_size, factor):
    kernel = np.random.rand(kernel_size, kernel_size) * factor
    return convolve(image, kernel)

"""
Functions to generate particle noise
"""

def speckle_noise(image, variance):
    noise = np.random.normal(0, np.sqrt(variance), image.shape)
    noisy_image = image + image * noise
    return noisy_image


def acoustic_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_signal = image + noise
    return noisy_signal


def thermal_noise(image, sigma_thermal):
    noise = np.random.normal(0, sigma_thermal, image.shape)
    noisy_signal = image + noise
    return noisy_signal


def electronic_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_signal = image + noise
    return noisy_signal

"""
Functions to generate artifacts 
"""

def ring_artifacts(image, ring_intensity, ring_frequency):
    x, y = np.meshgrid(np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[0]))
    ring_artifact = ring_intensity * np.sin(ring_frequency * (x ** 2 + y ** 2))
    noisy_image = image + ring_artifact
    return noisy_image

def clipping_artifacts():
    pass

def calibaration_artifacts(image, calibration_factor):
    noisy_signal = image * calibration_factor
    return noisy_signal

def shadow_artifacts(image, shadow_position, shadow_width, shadow_intensity):
    mask = np.ones(image.shape)
    start = max(shadow_position - shadow_width // 2, 0)
    end = min(shadow_position + shadow_width // 2, image.shape[1])
    mask[:, start:end] = 1 - shadow_intensity
    noisy_image = image * (1 - mask)
    return noisy_image

def reflection_artifact(image, noise_file, intensity_reduction):
    dot_sinogram = np.roll(noise_file, 50, axis=1)
    noise_sinogram = dot_sinogram * intensity_reduction
    final_sinogram = image + noise_sinogram
    return final_sinogram

# TODO: INPUT EXPLANATION
def ghost_artifacts(image, intensity_reduction, offset):
    # Creating a reflected version of the sinogram
    reflected_sinogram = image.copy()
    reflected_sinogram = np.roll(reflected_sinogram, offset, axis=1)
    reflected_sinogram = reflected_sinogram * intensity_reduction

    # Adding the reflected sinogram to the original
    noisy_sinogram = image + reflected_sinogram
    # return np.clip(noisy_sinogram, 0, 1)  # Clipping to maintain the original data range
    return noisy_sinogram