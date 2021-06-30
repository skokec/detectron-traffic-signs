import sys,time
import cv2

from echolib import pyecho
import echocv

class EcholibWrapper:
    def __init__(self, detection_method):
        self.camera = cv2.VideoCapture(0)

        self.loop = pyecho.IOLoop()
        self.client = pyecho.Client()
        self.loop.add_handler(self.client)

        self.enabled = False

        self.dockerCommandOut = pyecho.Publisher(self.client, sys.argv[1], "numpy.ndarray")
        self.dockerCommandIn = pyecho.Subscriber(self.client, sys.argv[2], "int", self.callback)

        self.detection_method = detection_method

    def callback(self, message):
        self.enabled = True if (pyecho.MessageReader(message).readInt() != 0) else False

    def run(self, wait_sec=10, sleep_sec=0):
        while self.loop.wait(wait_sec):
            _, frame = self.camera.read()

            writer = pyecho.MessageWriter()
            echocv.writeMat(writer, self.detection_method.doDetection(frame) if self.enabled else frame)
            self.dockerCommandOut.send(writer)

            if sleep_sec > 0:
                time.sleep(sleep_sec)
