import cv2
from imutils import paths
import os

def is_in_circle(x, y, centre_x, centre_y, radius):
    if (abs(x - centre_x) ** 2 + abs(y - centre_y) ** 2) ** (1 / 2) < radius:
        return True
    return False

def mask(image, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, name):
    centre_x = IMG_SIZE_WIDTH // 2
    centre_y = IMG_SIZE_HEIGHT // 2
    for i in range(IMG_SIZE_WIDTH):
        for j in range(IMG_SIZE_HEIGHT):
            if is_in_circle(i, j, centre_x, centre_y, IMG_SIZE_HEIGHT // 2):
                image[j,i, 3] = 255
            else:
                image[j,i, 3] = 0
    return image

IMG_SIZE_WIDTH = 150
IMG_SIZE_HEIGHT = 150
imagePaths = list(paths.list_images('Superheroes'))
print(f'{len(imagePaths)} images loaded.')

i = 1

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(    image, (IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT)    )
    # Adds a transparency channel to the image color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    # Making the images circled.
    image = mask(image, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, i)
    # Saving modified images.
    cv2.imwrite(f'images{os.sep}{i}.png', image)
    i += 1

print(f'The images are processed')
