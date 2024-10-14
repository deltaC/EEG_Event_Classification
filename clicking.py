import pygame
import time
from datetime import datetime

# Initialize pygame mixer
pygame.mixer.init()

# Load click sound
click_sound = pygame.mixer.Sound("Extra/click.wav")

# Define the interval between clicks (in seconds)
CLICK_INTERVAL = 1.15

# Store timestamps for each click
timestamps = ['Ear,Time']

# Start with the sound in the left ear
left_ear = True
counter = 0


while counter < 120:
    # Alternate between left and right ear
    channel = click_sound.play()
    if left_ear:
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
    
    # Alternate ears
    left_ear = not left_ear
    
    # Wait for the interval
    time.sleep(CLICK_INTERVAL)
    counter += 1


print("Program stopped.")

# Save timestamps to a file
with open("Extra/click_timestamps.csv", "w") as file:
    for timestamp in timestamps:
        file.write(timestamp + "\n")

print("Timestamps saved to click_timestamps.csv")