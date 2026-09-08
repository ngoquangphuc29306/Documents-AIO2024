import gradio as gr

def greet(name):
    return "Hello " + name + "!"
"""
with gr.Blocks() as demo:
    input = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")

    greet_btn = gr.Button("Submit")
    greet_btn.click(fn=greet, inputs=input, outputs=output)

demo.launch()
"""

# Change Event

"""
with gr.Blocks() as demo:
    input = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")

    input.change(fn=greet, inputs=input, outputs=output)

demo.launch()
"""

# Layout

def greet(first_name, last_name, age):
    output_1 = "Hello " + first_name + " " + last_name
    output_2 = "! You are " + age + " years old."
    return output_1 + output_2

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            first_name = gr.Textbox(label="First Name")
            last_name = gr.Textbox(label="Last Name")
        with gr.Column():
            age = gr.Textbox(label="Your Age")

    with gr.Row():
        output = gr.Textbox(label="Output Box")

    greet_btn = gr.Button("Submit")
    greet_btn.click(fn=greet,
                    inputs=[first_name, last_name, age],
                    outputs=[output])

demo.launch()