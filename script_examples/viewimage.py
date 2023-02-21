import cv2
import numpy as np
import gzip

def YUVtoRGB(byteArray, w, h):
    byteArray = np.frombuffer(byteArray, dtype=np.ubyte)
    e = w*h
    Y = byteArray[0:e]
    print(f"Reshaping {len(Y)} bytes into {w} by {h} pixels...")
    Y = np.reshape(Y, (h,w))
    Y = Y[:,:1920]

    s = e + int(e/4)
    V = byteArray[e:s]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (int(h/2),w))
    V = np.repeat(V, 2, 0)
    print(f"Reshaped V to {V.shape}")
    V = V[:,:1920]

    U = byteArray[s:]
    U = np.repeat(U, 2, 0)
    U = np.reshape(U, (int(h/2),w))
    U = np.repeat(U, 2, 0)
    U = U[:,:1920]
    print(f"Reshaped U to {U.shape}")

    RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
    return RGBMatrix

def crop(img, w, h):
    return

def decompress(buf):
    # work for both gzip and zlib
    # newbuf = zlib.decompress(buf, 15 + 16)
    newbuf = gzip.decompress(buf)
    print(f"BEFORE DECOMPRESS {len(buf)} | AFTER {len(newbuf)}")
    return newbuf

with open("captured1.yuv", "rb") as file:
    buf = file.read()
    buf = decompress(buf)


    w,h = 2048,1080

    print(f"Processing {len(buf)} bytes into {w} by {h} pixels...")

    img = YUVtoRGB(buf, w, h)

    print(f"img has shape: {img.shape}")

    # img = crop(buf, 1920, 1080)
    while True:
        cv2.imshow("Image", img)
        k = cv2.waitKey(33)
        if k==27:
            break
    cv2.imwrite("frame01.jpg", img)
