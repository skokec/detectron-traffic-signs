import time
import cv2

import echolib
from echolib.camera import Frame, FramePublisher, FrameSubscriber

from threading import Thread


class EcholibWrapper:
    
    def __init__(self, detection_method):

        self.loop   = echolib.IOLoop()
        self.client = echolib.Client()
        self.loop.add_handler(self.client)

        self.enabled = False

        self.docker_ready      = echolib.Publisher(self.client, "containerReady", "int")
        self.docker_command_in = echolib.Subscriber(self.client, "docker_demo_command_input", "int", self._docker_command_callback)
        
        self.camera_stream    = FrameSubscriber(self.client, "camera_stream_0", self._camera_stream_callback)
        self.docker_frame_out = FramePublisher(self.client, "docker_demo_output")

        self.detection_method = detection_method

        self.frame_in    = None
        self.frame_in_new = False

        self.frame_out    = None
        self.frame_out_new = False 

        self.closing = False

        self.n_frames = 0

    def _docker_command_callback(self, message):

        self.enabled = echolib.MessageReader(message).readInt() != 0

        print("Docker demo: got command {}".format(self.enabled))    

    def _camera_stream_callback(self, message):

        self.frame_in    = message.image
        self.frame_in_new = True

        self.n_frames += 1

        print("Docker demo: reading camera stream {}".format(self.n_frames))

    def process(self):
        
        while not self.closing:

            frame = None

            if self.enabled and self.frame_in_new:

                frame = self.frame_in
                self.frame_in_new = False
            
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = self.detection_method.predict(frame)

            if frame is not None:

                self.frame_out    = frame
                self.frame_out_new = True 
            
            time.sleep(0.01)
            
    def run(self, wait_sec=10, sleep_sec=0):

        for i in range(0,10):
            self.loop.wait(10)

            writer = echolib.MessageWriter()
            writer.writeInt(1)
            self.docker_ready.send(writer)

        thread = Thread(target = self.process)
        thread.start()

        print("Starting...")

        while self.loop.wait(1):

            print("In loop...")

            if self.frame_out_new:
                
                self.docker_frame_out.send(Frame(image = self.frame_out))
                self.frame_out_new = False 

        print("Stop")

        thread.join()

