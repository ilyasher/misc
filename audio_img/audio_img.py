#!/usr/bin/python3
import numpy as np
import sys
from scipy.io import wavfile
from PIL import Image

def encode_image(img_filename, audio_filename, new_img_filename):
    '''
    Combines an image and a .wav file into an image given by new_img_filename.
    new_img_filename should end ing .png so that full quality is preserved.
    The .wav file should be 16-bit with a 8000Hz sampling rate.
    '''
    # Load source image and sound
    rate, sound_data = wavfile.read(audio_filename)
    img = np.array(Image.open(img_filename))

    # If image is in Black and White, convert it into color
    if len(img.shape) == 2:
        # There's got to be a better way
        new_img = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            new_img[:, :, i] = img
        img = new_img
    height, width, _ = img.shape

    # Round width to the nearest even number
    img = img[:,:2*int(width/2),:]

    # Reduce resolution of image.
    # Last 2 out of 8 bits are cleared from each RGB value.
    img -= img % 4

    # Reduce sound from 16-bit to 12-bit
    # This is so that each value fits into 2 pixels, since each pixel (uint8)
    # now has 6 bits of free space.
    sound_data = np.right_shift(sound_data, 4)

    # Effectively we are pairing pixels up so that each value in the array has
    # free bits we can write to. Now one value from the sound_data can fit perfectly
    # into this array.
    img_reshaped = np.reshape(img, (height, int(width / 2), 6))

    # Reshape the sound array into the same size as the image.
    # If the sound data is larger than the image, shorten the audio clip.
    sound_resized = np.zeros(shape=img_reshaped.shape[0]*img_reshaped.shape[1], dtype=np.int16)
    if sound_data.shape[0] > sound_resized.shape[0]:
        sound_data = sound_data[:sound_resized.shape[0]]
    sound_resized[:sound_data.shape[0]] = sound_data
    sound_reshaped = np.reshape(sound_resized, img_reshaped.shape[:2])

    # Store 2 bits of the sound data into each RGB value.
    for i in range(6):
        audio_bits = (sound_reshaped % 4).astype(np.uint8)
        img_reshaped[:,:,i] += audio_bits
        sound_reshaped = np.right_shift(sound_reshaped, 2)

    # Reshape image to original size
    img = np.reshape(img_reshaped, img.shape)

    # Save image as a file
    Image.fromarray(img).save(new_img_filename)
    print("New image created:", new_img_filename)

def decode_image(img_filename, audio_filename=None, play_sound=False):
    '''
    Takes an image created with encode_image and extracts the sound from it into
    audio_filename. The audio file must end in .wav
    '''

    # Load source image
    img = np.array(Image.open(img_filename))

    # Keep only the last 2 bits of each pixel, and reshape image so that
    # each img[y, x] contains 12 bits of sound info.
    img = img % 4
    img = np.reshape(img, (img.shape[0], int(img.shape[1] / 2), 6))

    # Undo the encode_image process
    sound_data = np.zeros(shape=img.shape[:2], dtype=np.int16)
    for i in range(6)[::-1]:
        audio_bits = img[:,:,i]
        sound_data += audio_bits
        sound_data = np.left_shift(sound_data, 2)

    sound_data = np.left_shift(sound_data, 2)

    # Restore the sound to be shaped like a sequence
    sound_data = sound_data.flatten()

    # Save sound as a file
    rate = 8000
    if audio_filename:
        wavfile.write(audio_filename, rate, sound_data)

    if play_sound:
        import os
        from playsound import playsound
        tmp_filename = ".tmp.wav"
        wavfile.write(tmp_filename, rate, sound_data)
        try:
            playsound(tmp_filename)
        except KeyboardInterrupt:
            pass
        os.remove(tmp_filename)

if __name__ == "__main__":

    args = sys.argv
    exec = args[0]
    if len(args) == 2:
        decode_image(args[1], play_sound=True)
    elif len(args) == 3:
        decode_image(args[1], audio_filename=args[2])
    elif len(args) == 4:
        encode_image(args[1], args[2], args[3])
    else:
        print("Usage:")
        print("Encode audio into image:\n\t{} src_img src_audio.wav new_img.png".format(exec))
        print("Decode audio into file:\n\t{} src_img.png new_audio.wav".format(exec))
        print("Decode audio and play it:\n\t{} src_img.png".format(exec))
