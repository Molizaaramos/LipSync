import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno1.png')
    st.title('LipFusion')
    st.info('This application is originally developed from the LipSync deep learning model.')

st.title('LipSync Full Stack App')
# Dummy paths for video and alignment data
video_data_path = r'C:\path\to\dummy\video_data'  # Replace with actual path
alignment_data_path = r'C:\path\to\dummy\alignment_data'  # Replace with actual path

# Generating a list of options or videos
options = os.listdir(video_data_path)
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(video_data_path, selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video_file = open('test_video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', [frame.numpy().squeeze() for frame in video], fps=10)  # Ensure the frames are numpy arrays
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
