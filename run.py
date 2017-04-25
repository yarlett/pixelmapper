import cv2
import numpy as np
import pygame
import random
import time


from pixel_maps import get_locally_random_pixelmap, get_kaleidoscope_pixelmap
from PIL import Image


def get_random_pixelmap(k):
    """
    Returns a random pixelmap parameterized by an integer, 0-9 inclusive, passed in.
    """
    c = random.randint(10, 300)
    if k == 0:
        k = np.random.randint(10, 50)
        return dict(
            map=get_kaleidoscope_pixelmap(c, c, k),
            compute_size=(c, c),
        )
    elif k == 1:
        sigma = np.random.randint(1, 10)
        return dict(
            map=get_locally_random_pixelmap(c, c, sigma),
            compute_size=(c, c),
        )
    else:
        return dict(
            map=get_kaleidoscope_pixelmap(c, c, k),
            compute_size=(c, c),
        )


def get_webcam_image(camera, size):
    """
    Gets an image from the webcam using OpenCV as a (W x H x 3) numpy array.

    The image is resized to the specified size, and the channels are ordered to be in RGB order (they seem to get captured in BGR order for some reason).
    """
    ret, image = camera.read()
    image = Image.fromarray(image).resize(size, Image.ANTIALIAS)
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))
    image = np.array(image)
    return image


if __name__ == "__main__":
    # Parameters.
    FPS = 30                    # Frames per second.
    SCREEN_SIZE = (900, 900)    # Size of output images.
    SWITCH_SECONDS = 5.         # Change pixelmapper this frequently.

    # Initializations.
    CAMERA = cv2.VideoCapture(0)
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    pxmap = get_random_pixelmap(2)

    # Event loop.
    loop = True
    T = time.time()
    while loop:
        # Pygame event handling.
        clock.tick(FPS)
        for event in pygame.event.get():
            # Quit event.
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                loop = False
            # Numeric keypress event.
            if event.type == pygame.KEYDOWN:
                k = pygame.key.name(event.key)
                if k in "1234567890":
                    pxmap = get_random_pixelmap(int(k))

        # Get current image from the webcam.
        img = get_webcam_image(CAMERA, pxmap["compute_size"])

        # Apply kaleidoscopic map.
        img[pxmap["map"][:, 1], pxmap["map"][:, 0]] = img[pxmap["map"][:, 3], pxmap["map"][:, 2]]

        # Scale up to screen size.
        img = Image.fromarray(img)
        img = img.resize(SCREEN_SIZE, Image.ANTIALIAS)
        img = np.array(img)
        img = img.swapaxes(0, 1)

        # Display the new frame.
        pygame.display.set_caption("Pixemapper")
        pygame.surfarray.blit_array(screen, img)
        pygame.display.flip()

    # Release webcam resources.
    pygame.quit()
    CAMERA.release()
