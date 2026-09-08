# First app
import streamlit_demo as st

# Text element
st.title("This is my first Streamlit app!")
st.text("Streamlit is so easy!")

st.header("This is header")
st.header("This is header with divider", divider=True)
st.header(":blue[This is green header with divider]", divider=True)

st.header("AIO2024 Contest", divider=True)
st.subheader("Module 3 Examination", divider=True)

problem = """
### **Bài toán**
Cho một số nguyên không âm 'x', trả về căn bật hai của 'x' được làm tròn xuống thành số nguyên gần nhất. Số nguyên được trả về cũng phải là số không âm

### **Ví dụ**
#### Ví dụ 1:
- **Đầu vào**: 'x = 4'
- **Đầu ra**: '2'
- **Giải thích**: Căn bật hai của 4 là 2, vì vậy chúng ta trả về 2.
"""

st.markdown(problem)

code = """
class Solution(object):
    def mySqrt(self, x):
        return int(x**0.5)
"""

st.code(code, language="python")


# Input Element
st.title("This is INPUT ELEMENT")
name = st.text_input(
    label="Enter your name: ",
)

age = st.number_input(
    "Enter your age: ",
    value = 0,
    step = 1,
    format="%d"
)

age_1 = st.slider("How old are you?", 0, 60, 20)

st.write("Your name is: ", name)
st.write("Your age is: ", age)
st.write("I'm ", age_1, "years old")

is_student = st.checkbox("Are you student ?")
if is_student:
    st.write("Are you finish the contest?")

option = st.selectbox(
    "Which OS do you use yo learn Deep Learning?",
    ("Window", "Ubuntu", "MacOS"),
    index=None,
    placeholder="Select an OS...",
)

st.write("You choosed", option)

option_1 = st.radio(
    "Which OS do you use yo learn Deep Learning?",
    ["Windows", "Ubuntu", "MacOS"],
    captions=[
        "Windows good to Play games :video_game:",
        "Ubuntu good for learning CLI :smile:",
        "MacOS good for designer :vampire:",
    ]
)

st.write("You Choosed", option_1)

email = st.text_input(
    label="Type your email here: ",
    key="email"
)

if st.button("Submit"):
    if "@" not in email:
        st.write("Please type a valid email")
    else:
        st.write("Submitted")

# Media Element
st.title("This is Media Element")

from PIL import Image

uploaded_file = st.file_uploader("Choose an image...",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image.")

st.image("./love.png", caption="The logo app")

video_file = open("./60s.mp4", "rb") # rb = read-binary
video_bytes = video_file.read()

st.video(video_bytes)


# Layout Element
st.title("This is Layout Element")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Information")
    name_1 = st.text_input(
        label="Enter your name_1 : ", 
    )
    st.write("Your name is: ", name_1)
with col2:
    st.header("An image")
    st.image("./love.png")
with col3:
    st.header("An Video")
    video_file = open("./60s.mp4", "rb") # rb = read-binary
    video_bytes = video_file.read()
    st.video(video_bytes)

problem_1 = """
### **Bài toán**
Cho một số nguyên không âm 'x', trả về căn bật hai của 'x' được làm tròn xuống thành số nguyên gần nhất. Số nguyên được trả về cũng phải là số không âm

### **Ví dụ**
#### Ví dụ 1:
- **Đầu vào**: 'x = 4'
- **Đầu ra**: '2'
- **Giải thích**: Căn bật hai của 4 là 2, vì vậy chúng ta trả về 2.
"""

with st.container(height=450):
    st.markdown(problem_1)

# Status Element
st.title("Status Element")

email_1 = st.text_input(
    label="Type your email here: ",
    key="email1"
)

if st.button("Submit", key="submit_1"):
    if "@" not in email_1:
        st.error("Please type a valid email", icon="🆘")
    else:
        st.success("Submitted", icon="✅")

import time
import random

st.title("Load Machine Learning")
def load_model():
    with st.spinner('Model is being loaded...'):
        time.sleep(3)
    is_loaded = random.choice([True, False])

model = load_model()
if model:
    st.success("Model is loaded")
else:
    st.error('Model failed to load!')

