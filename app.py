"""Gradio entry point (also works as a Hugging Face Spaces app)."""
from agent_core import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
