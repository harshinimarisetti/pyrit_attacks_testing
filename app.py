"""
app.py
Simple Gradio question-answering UI, like a mini Gemini.
Run this cell in Colab AFTER model_utils has loaded the model.
Uses share=True since Colab has no accessible localhost.
"""

import gradio as gr
from model_utils import generate_answer


def answer_question(question: str) -> str:
    if not question or not question.strip():
        return "Please type a question."
    return generate_answer(question)


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question", placeholder="e.g. What is the capital of France?"),
    outputs=gr.Textbox(label="Answer"),
    title="Mini QA Assistant (Llama-3.2-3B)",
    description="Ask any question and get an answer.",
)

demo.launch(share=True, debug=False)
