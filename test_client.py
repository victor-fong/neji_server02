import socket
import time

HOST = "127.0.0.1"
PORT = 8989
SLEEP = 0.05

file = open("examples/recv_frame001.dat", "rb")
buf = file.read()
file.close()
frame_length = len(buf).to_bytes(4, byteorder="big")
content = frame_length + buf
print(f"CONTENT LENGTH {len(content)}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        s.sendall(content)
        time.sleep(SLEEP)
