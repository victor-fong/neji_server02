from status_worker import StatusWorker
from recent_queue import RecentQueue
from detector_torchvision import Detector
from frame_processor import FrameProcessor
from recv_socket import RecvSocket
import logging

PROCESSED_THREAD_NUM = 1


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    recv_queue = RecentQueue(1)
    processed_queue = RecentQueue(1)

    status_worker = StatusWorker()
    recv_socket = RecvSocket(status_worker, recv_queue)
    detector = Detector(processed_queue, status_worker)
    detector.start()
    for i in range(PROCESSED_THREAD_NUM):
        processor = FrameProcessor(f"worker{i}", recv_queue, processed_queue, status_worker)
        processor.start()
    status_worker.start()
    recv_socket.start()
