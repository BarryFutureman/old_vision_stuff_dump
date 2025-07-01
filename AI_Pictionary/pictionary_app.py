import numpy as np
import gradio as gr
import os
import json
import datetime

import app_theme

import sd_inference
# import realtime_whisper


def generate_image(prompt: str):
    output_file = "output_image.png"
    output_file = sd_model.predict(prompt)

    img = gr.Image(visible=True, type="pil", value=output_file)
    return img, gr.Image(visible=True, type="pil", value=output_file)


def load_image():
    output_file = "output_image.png"
    # img = gr.Image(visible=True, type="pil", value=output_file)

    return output_file


def show_answer():
    words = phrases[curr_phrase_index].split(" ")
    for w in words:
        correct_words.append(w)
    return gr.Markdown("## " + phrases[curr_phrase_index])


def next_phrase():
    global curr_phrase_index
    global correct_words

    if len(phrases)-1 > curr_phrase_index:
        curr_phrase_index += 1

    display_word_list = []

    for w in phrases[curr_phrase_index].split(" "):
        if w in correct_words:
            display_word_list.append(w)
        else:
            display_word_list.append("___")

    correct_words = []

    return f"## {phrases[curr_phrase_index]}.", gr.Markdown("## " + " ".join(display_word_list))


def show_word(text):
    global correct_words
    correct_words.append(text)

    return None


def add_phrase(text):
    phrases.append(text)

    return None


def check_answer():
    words = phrases[curr_phrase_index].split(" ")

    display_word_list = []
    if realtime_whisper_obj:
        text_results = realtime_whisper_obj.get_results()
    else:
        text_results = ""
    result_words = " ".join(text_results).split(" ")

    for w in words:
        for rw in result_words:
            if w.lower() == rw.lower():
                correct_words.append(w)

    for w in words:
        if w in correct_words:
            display_word_list.append(w)
        else:
            display_word_list.append("?"*len(w))

    return gr.Markdown("# " + "  ".join(display_word_list))


realtime_whisper_obj = None # realtime_whisper.RealtimeWhisper()
sd_model = sd_inference.LCMSDInfer()

phrases = ["banana dog",
           "dragon fruit",
           "bagel black hole",
           "Time travelers got Sam Altman fired by OpenAI to save the world from Skynet"]
correct_words = []
curr_phrase_index = 0

with gr.Blocks(title="AI Pictionary", theme=app_theme.Softy()) as demo:
    gr.Markdown("## 🎨 AI Pictionary")

    # Guess tab
    with gr.Tab("Guess"):
        profile_title_md = gr.Markdown(f"# Guess what this is?")
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                display_image = gr.Image(visible=True, value=None)
            with gr.Column(scale=1):
                pass
            # display_image.change(queue=True, fn=load_image, inputs=None, outputs=display_image, every=1)

        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                markdown = gr.Markdown("")
            with gr.Column(scale=1):
                pass

        with gr.Row():
            with gr.Accordion():
                refresh_button = gr.Button(value="refresh image", variant="secondary")
                refresh_button.click(fn=load_image, inputs=None, outputs=[display_image], every=1)
                whisper_button = gr.Button(value="refresh text", variant="secondary")
                whisper_button.click(fn=check_answer, inputs=None, outputs=[markdown], every=0.5)

    # Draw tab
    with gr.Tab("Draw"):
        with gr.Row():
            draw_tab_phrase_display = gr.Markdown(f"## {phrases[curr_phrase_index]}.")

        with gr.Row():
            # Input field
            text_box = gr.Textbox(label="Draw with words!")
            generate_button = gr.Button(value="Generate Image", variant="primary")
            generate_button.click(fn=generate_image, inputs=text_box, outputs=[gr.Image(), display_image])

        with gr.Row():
            with gr.Accordion():
                generate_button = gr.Button(value="Show Answer", variant="primary")
                with gr.Row():
                    show_part_text_box = gr.Textbox(label="Show a word")
                    show_part_button = gr.Button(value="Show", variant="secondary", size="sm")
                    show_part_button.click(fn=show_word, inputs=show_part_text_box, outputs=[show_part_text_box])
                generate_button.click(fn=show_answer, inputs=None, outputs=[markdown])

            with gr.Accordion():
                generate_button = gr.Button(value="Next", variant="primary")
                generate_button.click(fn=next_phrase, inputs=None, outputs=[draw_tab_phrase_display, markdown])
                with gr.Row():
                    add_new_text_box = gr.Textbox(label="New")
                    add_new_button = gr.Button(value="Add", variant="secondary", size="sm")
                    add_new_button.click(fn=add_phrase, inputs=add_new_text_box, outputs=[add_new_text_box])


    # demo.load(lambda: datetime.datetime.now(), None, None, every=1)

# realtime_whisper_obj = realtime_whisper.RealtimeWhisper()

demo.queue()
demo.launch(max_threads=8, server_port=5000, share=True)
