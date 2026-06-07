import time

print("Importing torchvision...")
t0 = time.time()
import torchvision
print(f"torchvision imported in {time.time()-t0:.2f}s")
try:
    print("Trying torchvision.io.read_video")
    v, a, info = torchvision.io.read_video("/home/team2/Unlearning/crema-d-mirror/VideoFlash/1001_DFA_ANG_XX.flv", pts_unit='sec')
    print("torchvision.io success! Video shape:", v.shape)
except Exception as e:
    print("torchvision.io failed:", e)

print("Importing librosa...")
t0 = time.time()
import librosa
print(f"librosa imported in {time.time()-t0:.2f}s")

print("Importing cv2...")
t0 = time.time()
import cv2
print(f"cv2 imported in {time.time()-t0:.2f}s")

