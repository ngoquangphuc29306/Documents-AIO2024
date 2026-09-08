import streamlit as st

st.image("./media/image.jpg", caption='This is a image',
         width=400)

st.audio('./media/audio.oga',start_time=100)
st.video('./media/video.mp4',start_time=3)