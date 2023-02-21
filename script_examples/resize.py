file = open("captured_frame1.dat", "rb")
buf = file.read()
file.close()

resize_level = 2

new_len = len(buf) / (resize_level * resize_level)

print(f"New Size {new_len}")
y_len = len(buf) / 4 * 2
u_len = len(buf) / 4
v_len = len(buf) / 4

y_buf = buf[:y_len]
u_buf = buf[y_len:y_len+u_len]
v_buf = buf[y_len + u_len :]
