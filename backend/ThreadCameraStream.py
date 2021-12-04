__all__=["CameraStream"]

# Import packages
import time
import cv2
from threading import Thread

class CameraStream:
    def __init__(self, is_ipUrl_: str, platform_: int, camera_source_: str,
    camera_id_: int, resolution_ipUrl=None, name_="CameraStream"):
        self.camera_id = camera_id_
        wait_time = 0
        self.not_connected = True
        self.isConnected = False
        self.is_ipUrl_ = is_ipUrl_
        while self.not_connected:
            time.sleep(1)
            # initialize the video camera stream and read the first frame
            # from the stream
            if is_ipUrl_:
                self.stream = cv2.VideoCapture(camera_source_)
            elif platform_ == 2: # windows
                self.stream = cv2.VideoCapture(camera_id_)
            elif platform_ == 1: # ubuntu
                # f'v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=(int)640, height=(int)480 ! 
                # nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink',
                self.stream = cv2.VideoCapture(camera_source_, cv2.CAP_GSTREAMER)
            else:
                raise Exception("Unexpected OS platform")    

            if self.stream.isOpened():
                self.not_connected = False
                self.isConnected = True
                print("===> \t CONNECTED... Camera", self.camera_id)
                break

            print("===> \t Reconnecting {}... Camera".format(wait_time), self.camera_id)
            wait_time += 1
            if wait_time == 10:
                print("===> \t Can not connect to the camera", self.camera_id)
                break
            
        if self.isConnected:
            if is_ipUrl_ and resolution_ipUrl is not None:
                self.stream.set(3,resolution_ipUrl[0])
                self.stream.set(4,resolution_ipUrl[1])
            print("camera", self.camera_id, "/t image width:", self.stream.get(3))
            print("camera", self.camera_id, "/t image height:", self.stream.get(4))

            (self.grabbed, self.frame) = self.stream.read()

            # initialize the thread name
            self.name = name_

            # initialize the variable used to indicate if the thread should
            # be stopped
            self.stopped = False
        else:
            self.stopped = True

    def start(self):
        if self.isConnected:
            # start the thread to read frames from the video stream
            t = Thread(target=self.update, name=self.name, args=())
            t.daemon = True
            t.start()
        return self

    def update(self):
		# keep looping infinitely until the thread is stopped
        while self.isConnected:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                break

			# otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.033)        # stream 30 FPS -> 1/30 ~ 0.0333 

    def read(self):
		# return the frame most recently read
       return self.grabbed, self.frame

    def getResolution(self):
        return self.stream.get(3), self.stream.get(4)   # W H

    def getFPS(self):
        return self.stream.get(5)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        self.stopped = True
        self.isConnected = False
        print("Release camera", self.camera_id)
        if not self.is_ipUrl_:      # No need release() for IP camera
            self.stream.release()

if __name__=="__main__":
    camera = CameraStream(
        is_ipUrl_=False,
        platform_=1,
        camera_source_=f'v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=(int)640, height=(int)480 ! nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink',
        camera_id_=0,
        resolution_ipUrl=None
    )

    (w, h) = camera.getResolution()
    print("W: {}  H: {}".format(w,h))
    fps = camera.getFPS()
    print("FPS: ", fps)

    camera.start()

    while camera.stream.isOpened():    #same OpenCV
        has, frame = camera.read()
        if has:
            cv2.imshow("TestStream", frame)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            break
    camera.release()
    