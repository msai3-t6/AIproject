#### IMPORTS ####################
import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio_and_save(save_path, n_times=50):
    """
    This function will run `n_times` and everytime you press Enter you have to speak the wake word

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
    """
    # Get the number of existing files
    start_number = len([f for f in os.listdir(save_path) if f.endswith('.wav')])

    input("To start recording Wake Word press Enter: ")
    for i in range(start_number, start_number + n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or to stop press ctrl + C ({i + 1}/{start_number + n_times}): ")

def record_background_sound(save_path, n_times=50):
    """
    This function will run automatically `n_times` and record your background sounds so you can make some
    keybaord typing sound and saying something gibberish.
    Note: Keep in mind that you DON'T have to say the wake word this time.

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
        Note: DON'T set it to the same directory where you have saved the wake word or it will overwrite the files.
    """
    # Get the number of existing files
    start_number = len([f for f in os.listdir(save_path) if f.endswith('.wav')])

    input("To start recording your background sounds press Enter: ")
    for i in range(start_number, start_number + n_times):
        fs = 44100
        seconds = 2 

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i+1}/{start_number + n_times}")

# Step 1: Record yourself saying the Wake Word
print("Recording the Wake Word:\n")
record_audio_and_save("audio_data/", n_times=100) 

# # Step 2: Record yourself saying the Wake Word
# print("Recording the Negative Word:\n")
# record_audio_and_save("negative_audio/", n_times=100) 

# # Step 2: Record your background sounds (Just let it run, it will automatically record)
# print("Recording the Background sounds:\n")
# record_background_sound("background_sound/", n_times=100)
