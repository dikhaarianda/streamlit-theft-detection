import cv2
import tempfile
import streamlit as st
from keras.models import load_model
from f1_score import EvaluationMetrics
from everyFrame_class import EveryFrame

def every_frame(load_class, sequence, model, video_file, audio_on):
  temp_file = tempfile.NamedTemporaryFile(delete=False)
  temp_file.write(video_file.read())

  cap = cv2.VideoCapture(temp_file.name)
  frame_idx = 0
  pencurian_count = 0
  frames = []
  show_frames = []
  is_playing = False

  predict_txt = 'Detecting...'
  pencurian_txt = 'Waiting...'
  normal_txt = 'Waiting...'

  video_col, text_col = st.columns([2, 1])
  with video_col:
    video_placeholder = st.empty()

  with text_col:
    frame_count = st.empty()
    pencurian_text = st.empty()
    normal_text = st.empty()
    predict_text = st.empty()
    attention_text = st.empty()
    audio = st.empty()

    if st.button('Stop Predict'):
      attention_text.empty()
      st.session_state.predict = False
      st.rerun()

  while(True):
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.resize(frame, (640, 480))
    frames.append(frame)
    frame_idx+=1

    if len(frames) == sequence:
      prediction = load_class.model_predict(frames, model)
      pencurian_txt = '{:.2f}%'.format(prediction[0][1] * 100)
      normal_txt = '{:.2f}%'.format(prediction[0][0] * 100)
      predict_txt = load_class.show_predict(prediction)

      if predict_txt == 'Pencurian':
        pencurian_count += 1
        if pencurian_count >= 3:
          show_frames.extend(frames)
          attention_text.warning('Attention, Theft Behavior Has Been Detected!')
          if audio_on and not is_playing:
            audio_file = open('assets/alarm_cut.mp3', 'rb')
            audio_file = audio_file.read()
            audio.audio(audio_file, format="audio/mp3", autoplay=True, loop=True)
            is_playing = True

      else:
        if audio_on:
          audio.empty()
        pencurian_count = 0
        attention_text.empty()
        is_playing = False

      frames = []
      frame_idx = 0

    video_placeholder.image(frame, channels='BGR')
    frame_count.text('Frame Count: ' + str(frame_idx+1))
    pencurian_text.text('Pencurian: ' + pencurian_txt)
    normal_text.text('Normal: ' + normal_txt)

    if predict_txt == 'Pencurian':
      predict_text.error('Predict: ' + predict_txt)
    elif predict_txt == 'Normal':
      predict_text.success('Predict: ' + predict_txt)
    else:
      predict_text.info('Predict: ' + predict_txt)
    cv2.waitKey(25)

  st.session_state.predict = False

  if show_frames:
    st.markdown('---')
    st.write('Report Frame:')

    img_show = 6
    column_num = 3
    row_num = 2
    frames_count = len(show_frames)
    skip_frames = max(frames_count // img_show, 1)

    column = st.columns(column_num)
    for idx_1 in range(row_num):
      for idx_2, col in enumerate(column):
        get_frame = (idx_1 * len(column) + idx_2) * skip_frames
        with col:
          st.image(show_frames[get_frame], channels='BGR', caption=f"Frame {get_frame+1}")

    st.info(f'total frame: {frames_count}')

def main():
  # frame preparation
  IMAGE_HEIGHT = 120
  IMAGE_WIDTH = 160
  SEQUENCE_COUNT = 30

  # model preparation
  model_path = 'assets/best_model.h5'
  model = load_model(model_path, custom_objects={'F1_Score': EvaluationMetrics.f1_score})

  # Program Title
  st.header('CCTV-based Theft Behavior Detection System')
  st.markdown('---')

  # setup sidebar
  st.sidebar.header('Navigation Pane')
  st.sidebar.markdown('---')
  audio_on = st.sidebar.toggle('Turn Alarm Sound', value=True)
  st.sidebar.markdown('---')
  upload_video = st.sidebar.file_uploader("Upload Your Video Here:", type=['mp4'])

  # video predictions
  if upload_video is None:
    st.sidebar.warning('Please select an options and upload the video before predictions!')
    st.session_state.predict = False
  else:
    st.sidebar.video(upload_video) # show video on sidebar

    if st.sidebar.button('Start Predict') or st.session_state.predict:
      st.session_state.predict = True
      ef = EveryFrame(IMAGE_WIDTH, IMAGE_HEIGHT) # call EveryFrame class for process video realtime
      every_frame(ef, SEQUENCE_COUNT, model, upload_video, audio_on)

# always run the file that is contained in the main
if __name__ == "__main__":
  main()