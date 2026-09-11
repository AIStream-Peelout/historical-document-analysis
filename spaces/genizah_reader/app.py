# File name: app.py
# Date: 9/10/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Genizah Reader — ZeroGPU Space serving qwen3-vl-8b-hebrew-v20a (merged bf16).

Three tasks with the exact prompts the model was trained on: transcribe a
page, transcribe line by line with boxes (drawn as they stream in), locate a
phrase. Model and processor are placed on CUDA at import time, as ZeroGPU
requires; each request runs inside ``@spaces.GPU`` with a task-sized budget.
"""
from threading import Thread

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from reader_core import (GPU_SECONDS, MAX_NEW_TOKENS, PROMPTS, draw_boxes, looping, parse_lines,
                         parse_locate, rtl)

MODEL_ID = "isaacmg/qwen3-vl-8b-hebrew-v20a-merged"
TASKS = {"Transcribe the page": "transcribe", "Lines with boxes": "lines", "Find a phrase": "locate"}
DISPLAY_MAX = 2000        # px, longest side of the annotated preview (boxes are resolution-free)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
model.eval()


class LoopStop(StoppingCriteria):
    """Stop generation when the decoded tail repeats itself (loop collapse)."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        tail = self.tokenizer.decode(input_ids[0, -256:], skip_special_tokens=True)
        return looping(tail)


def _duration(image, task_label, phrase) -> int:
    """Per-task GPU budget in seconds (shorter budgets get better queue priority)."""
    return GPU_SECONDS.get(TASKS.get(task_label, "lines"), 150)


def _preview(image: Image.Image) -> Image.Image:
    """Downscaled copy for drawing; the model always sees the full upload."""
    im = image.convert("RGB").copy()
    im.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    return im


@spaces.GPU(duration=_duration)
def read(image, task_label, phrase):
    """Stream the model's reading; yields (annotated preview, text, raw output)."""
    task = TASKS[task_label]
    if image is None:
        yield None, "Upload a page image first.", ""
        return
    phrase = (phrase or "").strip()
    if task == "locate" and not phrase:
        yield _preview(image), "Type the phrase to find (2–3 consecutive words work best).", ""
        return
    prompt = PROMPTS[task].format(phrase=phrase) if task == "locate" else PROMPTS[task]
    preview = _preview(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS[task],
                  do_sample=True, temperature=0.1, top_p=0.95,
                  stopping_criteria=StoppingCriteriaList([LoopStop(processor.tokenizer)]))
    Thread(target=model.generate, kwargs=kwargs, daemon=True).start()

    out = ""
    yield preview, "Reading the image…", ""
    for piece in streamer:
        out += piece
        if task == "lines":
            lines, _ = parse_lines(out)
            yield (draw_boxes(preview, [l["bbox_2d"] for l in lines]),
                   rtl("\n".join(l["text"] for l in lines)), out)
        elif task == "transcribe":
            yield preview, rtl(out), out
    if task == "locate":
        box = parse_locate(out)
        if box:
            yield draw_boxes(preview, [box], [phrase[:12]]), rtl(phrase), out
        else:
            yield preview, "The model did not return a box for that phrase.", out
    elif task == "lines":
        lines, complete = parse_lines(out)
        note = "" if complete else "\n\n(The reply was cut short — showing the lines that completed.)"
        if looping(out):
            note = "\n\n(The model started repeating itself and was stopped.)"
        yield (draw_boxes(preview, [l["bbox_2d"] for l in lines]),
               rtl("\n".join(l["text"] for l in lines)) + note, out)
    else:
        note = "\n\n(The model started repeating itself and was stopped.)" if looping(out) else ""
        yield preview, rtl(out) + note, out


BANNER = """
**Experimental research model — do not cite.** It reads handwritten Hebrew-script manuscripts
(Hebrew, Aramaic, Judaeo-Arabic) and is often right, but it also invents plausible words where the
page is damaged, sometimes repeats itself, and its boxes are approximate. The same image can read
slightly differently twice. Not trained on Arabic, Latin or Greek script. Uploads are not stored.
"""
FOOTER = """
Model: `isaacmg/qwen3-vl-8b-hebrew-v20a-merged` (Qwen3-VL-8B fine-tune). Training data credit:
the National Library of Israel's KTIV project and the Princeton Geniza Project, with images from the
holding institutions. Browse the collection at [cairogenizah.ai](https://cairogenizah.ai).
GPU time is provided by Hugging Face ZeroGPU and metered per visitor; sign in to Hugging Face for a larger daily quota.
"""

with gr.Blocks(title="Genizah Reader", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Genizah Reader")
    gr.Markdown(BANNER)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Manuscript page (JPEG/PNG)")
            task = gr.Radio(list(TASKS), value="Lines with boxes", label="Task")
            phrase = gr.Textbox(label="Phrase to find (for “Find a phrase”)", rtl=True,
                                placeholder="two or three consecutive words as written")
            run = gr.Button("Read", variant="primary")
        with gr.Column(scale=1):
            annotated = gr.Image(type="pil", label="Boxes", interactive=False)
            text = gr.Textbox(label="Reading", lines=14, rtl=True, text_align="right", interactive=False)
            with gr.Accordion("Raw model output", open=False):
                raw = gr.Textbox(lines=8, interactive=False, show_label=False)
    gr.Markdown(FOOTER)
    run.click(read, inputs=[image, task, phrase], outputs=[annotated, text, raw])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
