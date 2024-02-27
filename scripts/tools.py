import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys


def image_high_pass_filter(image, cutoff_freq):
    """Applique un filtre high-pass à une image

    Args:
        image (array): Image à filtrer
        cutoff_freq (int): Fréquence de coupure du filtre
    Returns:
       filtered_image (array): Image filtrée
    """

    # Realiser la transformée de Fourier
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Créer un masque pour le filtre high-pass
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0

    # Appliquer le masque
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    
    # Réaliser la transformée de Fourier inverse
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Convertir l'image en entiers non-signés
    filtered_image = np.uint8(img_back)
    
    return filtered_image


def image_low_pass_filter(image, cutoff_freq):
    """Applique un filtre low-pass à une image

    Args:
        image (array): Image à filtrer
        cutoff_freq (int): Fréquence de coupure du filtre
    Returns:
       filtered_image (array): Image filtrée
    """

    # Realiser un transformée de fourier
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Créer un masque pour le filtre low-pass
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1

    # Apliquer le masque
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    
    # Realiser un transformée de fourier inverse
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Convertir l'image en uint8
    filtered_image = np.uint8(img_back)
    
    return filtered_image


def filter_noise(bool_array):
    """Filter noise in a boolean array by converting sequences 
    of False of length <= 2 to True.

    Args:
        bool_array (array): array of boolean.

    Returns:
        bool_array (array): array of boolean with noise filtered.
    """
    sequences_to_convert = []

    # Detecter les sequences de False de longueur <= 2
    i = 0
    while i < len(bool_array):
        if not bool_array[i]:
            start = i
            while i < len(bool_array) and not bool_array[i]:
                i += 1
            end = i - 1
            if end - start <= 2:
                sequences_to_convert.append((start, end))
        else:
            i += 1

    # Convertir ces sequences detectés en True
    for start, end in sequences_to_convert:
        for j in range(start, end + 1):
            bool_array[j] = True

    return bool_array


def extract_band_coordinates(array):
    """Extract the start and end index of each band in a boolean array.

    Args:
        array (array): array of boolean.

    Returns:
        list: list of tuples containing the start and end index of each band.
    """
    # Initialiser les listes des indices de début et de fin de chaque bande
    start_indices = []
    end_indices = []

    # Iterer sur les elements de l'array
    for i, value in enumerate(array):
        # Verifier si la valeur courante est True et 
        # la valeur précédente est False (ou c'est le premier element)
        if value and (i == 0 or not array[i - 1]):
            start_indices.append(i)  

        # Verifier si la valeur courante est True et
        # la valeur suivante est False (ou c'est le dernier element)
        if value and (i == len(array) - 1 or not array[i + 1]):
            end_indices.append(i)  

    # Combiner les indices de début et de fin
    bands = list(zip(start_indices, end_indices))

    return bands 
