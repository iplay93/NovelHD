import numpy as np
import matplotlib.pyplot as plt

# 시간 변수 생성
t = np.linspace(0, 1, 1000)  # 0부터 1까지의 범위에서 1000개의 시간 점 생성

# 신호 생성
f1 = 10  # 주파수 10Hz
f2 = 25  # 주파수 25Hz
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)  # 10Hz와 25Hz의 주파수 성분을 가진 신호

# FFT 수행
fft_result = np.fft.fft(signal)

print(fft_result)