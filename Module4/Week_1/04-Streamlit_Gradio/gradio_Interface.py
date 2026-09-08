import gradio as gr

# Level 1: Interface


# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)


# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
#     api_name="predict"
# )

# demo.launch()

# def greet_1(first_name, last_name, age):
#     getting = "Hello, " + first_name + " " + last_name
#     if age < 18:
#         getting += ". You're a young!"
#     else:
#         getting += ". You're a man!"
#     return getting

# demo_1 = gr.Interface(
#     fn=greet_1,
#     inputs=['text', 'text', 'number'],
#     outputs=['text']
# )
# demo_1.launch()


# demo_2 = gr.Interface(
#     fn=greet_1,
#     inputs=['text', 'text', gr.Slider(1, 60, step=1)],
#     outputs=['text']
# )
# demo_2.launch()

# def greet_3(name, over_18, age):
#     getting = "Hello, " + name
#     if not over_18:
#         getting += ". You're a young!"
#     else:
#         getting += ". You're a man!"
#     return getting

# demo_3 = gr.Interface(
#     fn=greet_3,
#     inputs=['text', "checkbox"],
#     outputs=['text']
# )
# demo_3.launch()

def choice_device(demand):
    if demand == "Game":
        return "Windows"
    elif demand == "Deep Learning":
        return "Linux"
    else:
        return "MacOS"

# demo_4 = gr.Interface(
#     fn=choice_device,
#     inputs=[gr.Radio(["Game", "Deep Learning", "Web Development"])],
#     outputs=["text"]
# )  

# demo_4.launch()

import numpy as np
def filter(input_image):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_image.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo_5 = gr.Interface(
    fn=filter,
    inputs=["image"],
    outputs=["image"]
)
demo_5.launch()



