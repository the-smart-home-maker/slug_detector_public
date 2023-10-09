import io
import time
from PIL import Image
from picamera import PiCamera
import RPi.GPIO as GPIO
import time
import numpy as np
import math
import requests
import json
from json import JSONEncoder
from ftplib import FTP

TF_SERVING_URL = "" # URL OR IP-ADDRESS TO YOUR TF SERVING SERVER
TF_SERVING_PORT = "" # PORT OF YOUR TF SERVING SERVER
LOCAL_IMAGE_FOLDER = "/home/pi/Pictures/slug_detector/" # FOLDER IN WHICH IMAGES SHOULD BE STORED LOCALLY FOR DEBUGGING PURPOSES
HASS_URL = "" # URL OF YOUR HOME ASSISTANT INSTANCE
FTP_USER = "" # FTP USER SET UP IN HOME ASSISTANT FTP ADD-ON
FTP_PASS = "" # FTP PASSWORD SET UP IN HOME ASSISTANT FTP ADD-ON
HASS_WEBHOOK_URL = "" # URL OF HOME ASSISTANT WEBHOOK WHICH WILL TRIGGER AUTOMATION
TURNS_PER_DIRECTION = 3 # HOW MANY PANS (OF HOUSING) SHALL BE MADE IN EACH DIRECTION
IMG_SIZE = (192, 192) # DEFINE IMAGE SIZE

# set GPIOs (this needs to be adjusted to the GPIOs you selected in the wiring step)
gpio_coil_A1 = 2
gpio_coil_A2 = 3
gpio_coil_B1 = 4
gpio_coil_B2 = 17

# don't lower this too much as your stepper motor might not be capable of handling a value too low
step_sleep = 0.005 

# 4096 steps is 360°
# I want one step to be 30° which is 1/12 of 360°
step_count = int(4096 / 12) 

# defining stepper motor sequence
step_sequence = [[1,0,0,1],
                 [1,0,0,0],
                 [1,1,0,0],
                 [0,1,0,0],
                 [0,1,1,0],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,0,0,1]]

# setting up GPIOs
GPIO.setmode( GPIO.BCM )
GPIO.setup( gpio_coil_A1, GPIO.OUT )
GPIO.setup( gpio_coil_A2, GPIO.OUT )
GPIO.setup( gpio_coil_B1, GPIO.OUT )
GPIO.setup( gpio_coil_B2, GPIO.OUT )

# initializing GPIOs
GPIO.output( gpio_coil_A1, GPIO.LOW )
GPIO.output( gpio_coil_A2, GPIO.LOW )
GPIO.output( gpio_coil_B1, GPIO.LOW )
GPIO.output( gpio_coil_B2, GPIO.LOW )

motor_pins = [gpio_coil_A1,gpio_coil_A2,gpio_coil_B1,gpio_coil_B2]

# pull all GPIOs low
def cleanup():
    GPIO.output( gpio_coil_A1, GPIO.LOW )
    GPIO.output( gpio_coil_A2, GPIO.LOW )
    GPIO.output( gpio_coil_B1, GPIO.LOW )
    GPIO.output( gpio_coil_B2, GPIO.LOW )
    GPIO.cleanup()

def turn(left=False):
    i = 0
    motor_step_counter = 0
    for i in range(step_count):
        for pin in range(0, len(motor_pins)):
            GPIO.output( motor_pins[pin], step_sequence[motor_step_counter][pin] )
        if left == True:
            motor_step_counter = (motor_step_counter - 1) % 8
        elif left == False:
            motor_step_counter = (motor_step_counter + 1) % 8
        else: # defensive programming
            print( "wrong data format of parameter left" )
            cleanup()
            exit( 1 )
        time.sleep( step_sleep )

def turnLeft():
    turn(True)

def turnRight():
    turn()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class_names = ['no_slug', 'slug']
goLeft = False
count_turns = 0

try:
    camera = PiCamera()
    camera.resolution = IMG_SIZE
    camera.start_preview()
    camera.rotation = 270
    time.sleep(2)
    stream = io.BytesIO()
    while(True):
        
        # get image from PI camera
        camera.capture(stream , format='jpeg')
        stream.seek(0)
        img = Image.open(stream)
        img.load()

        # prepare image and as numpy array required by TF serving
        img_array = np.asarray( img, dtype="int32" )
        img_batch = np.expand_dims(img_array, axis=0)
        
        # call TF serving server to get prediction
        url = "http://" + TF_SERVING_URL + ":" + TF_SERVING_PORT + "/v1/models/slug_detector:predict"
        result = requests.post(url, data='{"instances": ' + json.dumps(img_batch, cls=NumpyArrayEncoder) + '}', timeout=20)
        result_json = result.json()
        x = np.linspace(-10, 10, 100)
        z = 1/(1 + np.exp(-result_json["predictions"][0][0]))

        # if z < 0.5, then we say no slug has been detected. Otherwise, a slug has been detected.
        prediction = 1
        if z < 0.5:
            prediction = 0

        # save image locally on the Raspberry pi including prediction in name. This helps for later debugging and fine tuning your model.
        img_filename = str(time.time()) + ".jpg"
        img_path = LOCAL_IMAGE_FOLDER + class_names[prediction] + "/" + img_filename
        img.save(img_path)

        # if slug has been detected
        if (prediction == 1):
            # upload image to Home Assistant via FTP
            ftp = FTP(host=HASS_URL)
            ftp.login(user=FTP_USER, passwd=FTP_PASS)
            ftp.cwd('config')
            ftp.cwd('www')
            file=open(img_path, 'rb')
            ftp.storbinary('STOR ' + img_filename, file)
            file.close()
            ftp.quit()

            # now send "slug found" event to Home Assistant via webhook
            url = HASS_WEBHOOK_URL
            r = requests.post(url, json={"filename": img_filename})

        stream.seek(0)
        stream.truncate()
        time.sleep(1)
        if goLeft:
            turnLeft()
        else: 
            turnRight()
        count_turns = count_turns + 1
        if count_turns % TURNS_PER_DIRECTION == 0:
            goLeft = not goLeft
            
finally:
    print("done")
    GPIO.cleanup() # cleanup all GPIO 
