import streamlit as st

image_list = ['media/amblem.png', 
              'media/youtube.png']
caption_list = ['Amblem', 'Youtube']
st.header('Welcome to Peter Phuc')
st.image(image=image_list, width=100, caption=caption_list)
st.subheader('Peter Phuc is a Student and\
             he is studying Streamlit')
st.link_button('Go to Facebook',
               'https://www.facebook.com/ngoquangphuc29306/')