import imageio
import os
import numpy as np

data = np.load('decay_c_36x36x20000x4x4x2x0.4x1.0x5.0x-1.0x-1.0x1.0x25.0.npy') # decay_ or not

# Create a list of filenames for 2D concentration field figures
filenames = [f'decay_2D_concentration_field_{i}.png' for i in range(0, len(data[0, 0, :]), 10)] # decay_ or not

# Create a GIF from the saved figures
with imageio.get_writer('decay_2D_concentration_field.gif', mode='I') as writer: # decay_ or not
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        print(f"Added {filename} to the GIF")
print("GIF creation done!")
