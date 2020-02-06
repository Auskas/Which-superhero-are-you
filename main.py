
from kivy import platform
from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger
from imutils import paths
import numpy as np
import cv2
import PIL
import os
import random
import time

if platform == 'android':
    # Lines below are used for vibrator functionality.
    from jnius import autoclass, cast
    from plyer.platforms.android import SDK_INT
    from plyer.platforms.android import activity
    from vibrator import Vibrator

    # Accelerometer using plyer.
    from plyer import accelerometer
    accelerometer.enable()

    # Checks required Android permissions. Camera access has to be granted prior to the main code.
    from android.permissions import check_permission, request_permissions, Permission
    if check_permission.CAMERA == False and check_permission.VIBRATE == False:
        request_permissions([Permission.CAMERA, Permission.VIBRATE])
    while check_permission.CAMERA == False and check_permission.VIBRATE == False:
        time.sleep(1)

# Gets full images paths for images in 'images' folder.
imagePaths = list(paths.list_images('images'))
NUMBER_OF_IMAGES = len(imagePaths)
Logger.info(f'{NUMBER_OF_IMAGES} images loaded.')
random.seed(42)
random.shuffle(imagePaths)
superheroes = []

# Loads the images, resizes them and appends to a list.
for imagePath in imagePaths:
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (300,300))
    superheroes.append(image)
IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, _ = superheroes[0].shape
Logger.info(f'The images are processed')

FPS = 25 # desired number of frames per second.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
minW = 128
minH = 96

class IMG(Image):

    def __init__(self, **kwargs):
        self.orientation = 0
        self.exposure_time = time.time() # initial time 
        self.i = 0 # for iterating over superheroes images.
        if platform == 'android':
            self.vibrator = Vibrator() 
        self.target_time = random.uniform(3, 4.2) # a random float number of seconds to look at the screen.
        super(Image, self).__init__(**kwargs)

    def update(self, texture):
        """ Gets the texture of a frame. Converts it to a numpy array,
            calls frame_processing method and converts the processed image(array)
            back to the texture."""
        # Converts OpenGL texture into RGB array.
        image = PIL.Image.frombytes(mode='RGBA',
            size=(int(texture.size[0]), int(texture.size[1])),
            data=texture.pixels).convert('RGB')
        # Converts the array into a numpy array.
        image = np.array(image)

        # Flips the image around the Y-axis.
        image = cv2.flip(image, 1)
        if platform == 'android':
            self.get_orientation()
        if self.orientation == 1:
            image = cv2.flip(image, 0)
            cv2.putText(image, 'Please hold your device vertically', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2)
            image = cv2.flip(image, 0)
        else:
            image = self.frame_processing(image)           
        buf = image.tostring()
        texture1 = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = texture1
        # The try/except block is used for performance evaluation. It calculates real FPS.
        try:
            frame_time = time.time() - self.start_time
            current_FPS = round(1 / frame_time, 0)
            Logger.info(f'{current_FPS} FPS')
        except Exception as error:
            Logger.info('First frame.')
        self.start_time = time.time()

    def frame_processing(self, frame):
        """ Method for processing each frame in order to detect faces and draw the superhero widget.
            Gets a frame, creates a copy of the grayscaled frame, detects faces in the frame.
            Draws the superhero widget using the coordinates of the bounding box of a face.
            Counts number of seconds passed since the face is detected and stops the widget update
            as it reaches predetermind value. Returns the processed frame.""" 
        if platform == 'android':
            # Rotates the frame 90 degrees clockwise in order to get the portrait mode.
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
                                            gray,
                                            scaleFactor = 1.3,
                                            minNeighbors = 5,
                                            minSize = (int(minW), int(minH)),
                                            )
        # If the screen is untouched and less than random.uniform(3, 4.2) seconds passed since face detection,
        # draws the next superhero from the list.
        if cameraExample.touch.touch:
            self.i += 1
            # If superhero iterator index is bigger than the number of superheroes, it is reset to zero.
            if self.i > NUMBER_OF_IMAGES - 1:
                self.i = 0
        else:
            if platform == 'android':
                # Frames are mirrored, flip them to put the message.
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, 'Touch screen to restart', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2)
                frame = cv2.flip(frame, 1)
            else:
                cv2.putText(frame, 'Click screen to restart', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2)
            self.exposure_time = time.time()
        j = 0 # this index is used to draw another superhero if multiple faces are detected.
        # x,y are coordinates of the left top corner of the bounding box of a face.
        # w and h are width and height of the bounding box.
        for(x,y,w,h) in faces: 
            if time.time() - self.exposure_time > self.target_time:
                cameraExample.touch.change() # in order to stop iterating over superheroes.
                if platform == 'android':
                    self.vibrator.vibrate(1000) # vibrate for 1 second.
            if self.i + j > len(superheroes) - 1: # if superhero index is greater than the size of the list.
                next_superhero = 0
            else:
                next_superhero = self.i + j
            superhero = superheroes[next_superhero]
            b,g,r,a = cv2.split(superhero) # gets independent arrays for color channels as well as the alpha.
            overlay_color = cv2.merge((b,g,r))
            # The mask is created to overlay the frame with the widget. medianBlur is used for smooth edges.
            mask = cv2.medianBlur(a,5) 

            # Draws a blue circle on top of the bounding box. A superhero will be drawn inside.
            circle_centre = (x + w // 2, y - IMG_SIZE_HEIGHT // 2)
            cv2.circle(frame, circle_centre , IMG_SIZE_HEIGHT // 2, [0, 0, 255], thickness = 10)

            # Adjusts the leftmost coordinate of the widget.
            # (Widget could be bigger or smaller than the bounding box.)
            if w > IMG_SIZE_WIDTH:
                x += int((w - IMG_SIZE_WIDTH) / 2)
            elif w < IMG_SIZE_WIDTH:
                x -= int((IMG_SIZE_WIDTH - w) / 2)
            
            # Checks if the superhero widget if partially off the screen.
            # First checks the horizontal position of the widget.
            if x < 0: # the left side of the widget is off the screen.
                mask_begin_x, mask_end_x = -x, IMG_SIZE_WIDTH
                roi_begin_x, roi_end_x = 0, IMG_SIZE_WIDTH + x
            elif x + IMG_SIZE_WIDTH > frame.shape[1]: # the right side of the widget is off the screen.
                mask_begin_x, mask_end_x = 0, frame.shape[1] - x - IMG_SIZE_WIDTH
                roi_begin_x, roi_end_x = x, frame.shape[1]
            else: # widget is entirely on screen.
                mask_begin_x, mask_end_x = 0, IMG_SIZE_WIDTH
                roi_begin_x, roi_end_x = x, x + IMG_SIZE_WIDTH

            # Checks the vertical position of the widget.
            if y < IMG_SIZE_HEIGHT: # widget top is off the screen.
                mask_begin_y, mask_end_y = IMG_SIZE_HEIGHT - y, IMG_SIZE_HEIGHT
                roi_begin_y, roi_end_y = 0, y
            else: # widget is entirely on screen.
                mask_begin_y, mask_end_y = 0, IMG_SIZE_HEIGHT
                roi_begin_y, roi_end_y = y - IMG_SIZE_HEIGHT, y

            # Adjusts mask, overlay_color according to the previous conditions.
            # Mask and overlay_color are cut if the widget is partially off screen.
            mask = mask[mask_begin_y: mask_end_y, mask_begin_x : mask_end_x]
            overlay_color = overlay_color[mask_begin_y: mask_end_y, mask_begin_x : mask_end_x]
            # Region of interest of the frame to overlay withe the current superhero image.
            roi = frame[roi_begin_y : roi_end_y, roi_begin_x : roi_end_x]

            # Pixels under transparent area of superhero images are left untouched.
            img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))
            # Pixels under non-transparent area of superhero images are replaced with the image pixels.
            img2_fg = cv2.bitwise_and(overlay_color,overlay_color, mask = mask)
            frame[roi_begin_y : roi_end_y, roi_begin_x : roi_end_x] = cv2.add(img1_bg, img2_fg)

            j += 1 # next superhero index for the next detected face.

        if len(faces) == 0:
            # Resets the time counter if there is no face detected.
            self.exposure_time = time.time()
            self.target_time = random.uniform(3, 4.2)
            if platform == 'android':
                frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'Please look at the screen', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2)
            if platform == 'android':
                frame = cv2.flip(frame, 1)
        if platform == 'android':
            # Rotates the frame in the opposite direction to make it landscape again.
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame = cv2.flip(frame, 0)
        return frame

    def get_orientation(self):
        """ Gets current device orientation using accelerometer data.
            If the device accelerometer x coordinate indicates that it is portrait,
            self.orientation = 0, otherwise, self.orientation = 1."""
        if -4.9 < accelerometer.acceleration[0] < 4.9:
            self.orientation = 0
        else:
            self.orientation = 1

class CameraExample(App):

    def build(self):
        self.layout = FloatLayout()
        self.touch = Foo()
        # Create a camera object. Use index=0 for rear camera and index=1 for front camera.
        self.cameraObject = Camera(index=0, resolution=(960,540), play=True)
        self.display = IMG()
        self.layout.add_widget(self.display)
        self.layout.add_widget(self.touch) # we have to add that object to the layout to detects touches.
        Logger.info(f'Schedule interval is {round(1 / FPS, 3)} seconds')
        Clock.schedule_interval(self.update, round(1 / FPS, 3))
        return self.layout

    def update(self, t):
        if self.cameraObject.texture is None:
            Logger.warning('No Feed!')
        else:
            self.display.update(self.cameraObject.texture)

# A dummy class which only purpose to detects screen touches and change its status accordingly.
class Foo(FloatLayout):
    def __init__(self, **kwargs):
        self.touch = True
        super(FloatLayout, self).__init__(**kwargs)

    def on_touch_down(self, t):
        if self.touch:
            self.touch = False
        else:
            self.touch = True

    def change(self):
        if self.touch:
            self.touch = False
        else:
            self.touch = True

if __name__ == '__main__':
    cameraExample = CameraExample()
    cameraExample.run()    

__version__ = '0.1'