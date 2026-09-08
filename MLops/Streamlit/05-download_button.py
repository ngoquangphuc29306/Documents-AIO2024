import streamlit as st

st.image('./media/image.jpg', caption='This is a image')
file_name = st.text_input('Enter file name')
st.write(file_name)

with open('image.jpg', 'rb') as file:
    btn = st.download_button(
        label="Download Image",
        data=file,
        file_name=file_name,
        mime="image/png"
    )