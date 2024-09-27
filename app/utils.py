import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)  # Grayscale
        frame = frame[190:236, 80:220]  # Crop
        frames.append(frame.numpy())  # Convert to numpy and store
    cap.release()
    
    frames = tf.convert_to_tensor(frames, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    
    normalized_frames = (frames - mean) / std  # Normalize the frames
    
    # Convert the normalized frames back to uint8 (0-255 range)
    frames_uint8 = tf.clip_by_value((normalized_frames * 255), 0, 255)  # Rescale and clip
    frames_uint8 = tf.cast(frames_uint8, tf.uint8)  # Convert to uint8
    
    return frames_uint8

def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]
    # Dummy paths for video and alignment data
    video_path = os.path.join(r'C:\path\to\dummy\video_data', f'{file_name}.mpg')  # Replace with actual path
    alignment_path = os.path.join(r'C:\path\to\dummy\alignment_data', f'{file_name}.align')  # Replace with actual path
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments
