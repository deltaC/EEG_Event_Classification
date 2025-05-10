import pygame
import time
import numpy as np
from datetime import datetime

# Initialize pygame mixer
pygame.mixer.init()

# Load click sound
click_sound = pygame.mixer.Sound("Extra/click.wav")

# Define the interval between clicks (in seconds)
N_CLICKS = 62
click_intervals = 1.2 + np.random.rand(N_CLICKS)

# Store timestamps for each click
timestamps = ['Ear,Time']

# Start with the sound in the left ear
is_left_ear = [i < 0.5 for i in np.random.rand(N_CLICKS)]


for i in range(N_CLICKS):
    # Alternate between left and right ear
    channel = click_sound.play()
    if is_left_ear[i]:
        # Set volume to play in the left ear
        channel.set_volume( 1.0, 0.0)  # Left ear, no sound in right
        ear = "Left"
    else:
        # Set volume to play in the right ear
        channel.set_volume(0.0, 1.0)  # Right ear, no sound in left
        ear = "Right"
    
    # Get the current timestamp and save it
    current_time = datetime.now()
    timestamps.append(f"{ear},{current_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
    print(timestamps[-1])
    
    # Wait for the interval
    time.sleep(click_intervals[i])


print("Program stopped.")

# Save timestamps to a file
with open("click_timestamps2.csv", "w") as file:
    for timestamp in timestamps:
        file.write(timestamp + "\n")

print("Timestamps saved to click_timestamps2.csv")