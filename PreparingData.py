#### IMPORTS ####################
import sounddevice as sd
from scipy.io.wavfile import write

def record_etc(save_path, n_times=50):
    input("To start recording etc press Enter: ")
    
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_gaesaekki(save_path, n_times=50):
    input("To start recording gaesaekki press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_shibal(save_path, n_times=50):
    input("To start recording shibal press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

# # Step 1: Record etc
# print("Recording etc:\n")
# record_etc("./myproject/etc/", n_times=100) 

# # Step 2: Record gaesaekki
# print("Recording gaesaekki:\n")
# record_gaesaekki("./myproject/gaesaekki/", n_times=100)

# Step 3: Record shibal
print("Recording shibal:\n")
record_shibal("./myproject/shibal/", n_times=100)