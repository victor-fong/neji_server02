import socket
import threading
import cv2
import queue
import time
import os
import gzip
import numpy as np
import logging
from imageai.Detection import ObjectDetection
# import gzip
# from io import BytesIO

SAVE = True
MAX_SAVE = 20

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

HOST = "0.0.0.0"  # Standard loopback interface address (localhost)
PORT = 8989  # Port to listen on (non-privileged ports are > 1023)

def YUVtoRGB(byteArray, w, h):
    byteArray = np.frombuffer(byteArray, dtype=np.ubyte)
    e = w*h
    Y = byteArray[0:e]
    Y = np.reshape(Y, (h,w))
    # Y = Y[:,:1920]

    s = e + int(e/4)
    V = byteArray[e:s]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (int(h/2),w))
    V = np.repeat(V, 2, 0)
    # V = V[:,:1920]

    U = byteArray[s:]
    U = np.repeat(U, 2, 0)
    U = np.reshape(U, (int(h/2),w))
    U = np.repeat(U, 2, 0)
    # U = U[:,:1920]

    RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
    return RGBMatrix

def crop(img, w, h):
    return

def decompress(buf):
    # work for both gzip and zlib
    # newbuf = zlib.decompress(buf, 15 + 16)
    newbuf = gzip.decompress(buf)
    return newbuf


next_frame = None
next_frame_cond = threading.Condition()

status_lock = threading.Lock()
recv_counter = 0
processed_counter = 0
detected_counter = 0

detection_queue = queue.Queue(maxsize=10)

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
# do one to initialize detector
detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "frame01.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)
logging.info("DETECTOR Initialized")

def start_detection():
    global detector, status_lock, detected_counter, detection_queue
    while True:
        img = detection_queue.get()
        logging.info("DETECTING")
        returned_image, detections = detector.detectObjectsFromImage(input_type="array", input_image=img, output_type="array", minimum_percentage_probability=20, thread_safe=True)
        logging.info(f"DETECTED {len(detections)} objects")
        status_lock.acquire()
        detected_counter += 1
        status_lock.release()



def start_processor():
    global next_frame, next_frame_cond, status_lock, processed_counter, detection_queue, counter

    while True:
        buf = None
        with next_frame_cond:
            if next_frame == None:
                next_frame_cond.wait()
            buf = next_frame
            next_frame=None
        if buf == None:
            continue
        buf = decompress(buf)
        w = int.from_bytes(buf[0:4], "big")
        h = int.from_bytes(buf[4:8], "big")
        buf = buf[8:]
        
        img = YUVtoRGB(buf, w, h)
        if SAVE:
            jpg_filename = f"work/processed_frame{counter:03d}.jpg"
            cv2.imwrite(jpg_filename, img)
            detected_filename = f"work/detected_frame{counter:03d}.jpg"
            detections = detector.detectObjectsFromImage(input_type="array", input_image=img, output_image_path=detected_filename, output_type="file", minimum_percentage_probability=20, thread_safe=True)
        else:
            returned_image, detections = detector.detectObjectsFromImage(input_type="array", input_image=img, output_type="array", minimum_percentage_probability=20, thread_safe=True)
        # detection_queue.put(img)

        status_lock.acquire()
        processed_counter += 1
        status_lock.release()


        # print(f"img has shape: {img.shape}")

        # img = crop(buf, 1920, 1080)
        # while True:
        #     cv2.imshow("Image", img)
        #     k = cv2.waitKey(33)
        #     if k==27:
        #         break




def status_thread():
    global status_lock, recv_counter, processed_counter, detected_counter
    old_time = round(time.time()*1000)
    while True:
        time.sleep(5)
        status_lock.acquire()
        new_time = round(time.time()*1000)
        cdetected_counter = detected_counter
        crecv_counter = recv_counter
        cprocessed_counter = processed_counter
        recv_counter = 0
        processed_counter = 0
        detected_counter = 0
        status_lock.release()
        time_span = float(new_time - old_time) / 1000.0
        recv_fps = crecv_counter / time_span
        process_fps = cprocessed_counter / time_span
        detected_fps = cdetected_counter / time_span
        old_time = new_time
        logging.info(f"RECEIVED {recv_fps:5.4f} FPS | PROCESSED {process_fps:5.4f} FPS | DETECTED {detected_fps:5.4f} FPS | TIME SPAN {time_span}")

def start_receiver():
    global next_frame, next_frame_cond, recv_counter, status_lock, counter
    counter = 0
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            logging.debug("Waiting for Connection...")
            conn, addr = s.accept()
            with conn:
                try:
                    logging.debug(f"Connected by {addr}")
                    while True:
                        data = conn.recv(4)
                        if not data:
                            break
                        frameLength = int.from_bytes(data, "big")
                        logging.debug(f"Receiving {frameLength} bytes of frame data")
                        chunks = []
                        bytes_read = 0
                        while bytes_read < frameLength:
                            chunk = conn.recv(min(frameLength - bytes_read, 4096))
                            if chunk == b'':
                                raise RuntimeError("socket connection broken")
                            chunks.append(chunk)
                            bytes_read = bytes_read + len(chunk)
                        chunks = b''.join(chunks)
                        if SAVE:
                            counter += 1
                            if counter > MAX_SAVE:
                                counter = 0
                            file = open(f"work/recv_frame{counter:03d}.dat", "wb")
                            file.write(chunks)
                            file.close()

                        next_frame = chunks
                        status_lock.acquire()
                        recv_counter += 1
                        status_lock.release()
                        with next_frame_cond:
                            next_frame_cond.notify()
                except:
                    logging.error("Connection reset")

                    # print(f"Finished receiving {frameLength} bytes of frame data")
                    # # inbuffer = BytesIO(chunks)
                    # counter += 1
                    # file = open(f"captured{counter}.yuv", "wb")
                    # file.write(chunks)
                    # file.close()

receiver_thread = threading.Thread(target=start_receiver)
receiver_thread.start()

detection_thread_count = 1
for i in range(detection_thread_count):
    processor_thread = threading.Thread(target=start_processor)
    processor_thread.start()

status_thread = threading.Thread(target=status_thread)
status_thread.start()

for i in range(detection_thread_count):
    detection_thread = threading.Thread(target=start_detection)
    detection_thread.start()




# while True:
#     k = cv2.waitKey(33)
#     if k==27:
#         break
