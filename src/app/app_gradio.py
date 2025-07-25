import gradio as gr
import os


# --- Core Functions ---

def summarize(dialogue_text: str) -> str:
    """
    Placeholder function to simulate summarization.
    This function should be replaced with your actual model inference logic.
    """
    print(f"Received for summarization: {dialogue_text}")
    # Replace this with your model's prediction logic
    if not dialogue_text:
        return "Please enter some text to summarize."
    
    # Simple transformation as a placeholder
    summary = f"Summary of: '{dialogue_text[:30]}...'"
    
    print(f"Generated summary: {summary}")
    return summary

# --- Gradio Interface ---

def create_gradio_app():
    """Creates and returns the Gradio web interface."""
    
    example_dialogue = (
        "#Person1#: 안녕하세요, 오늘 날씨가 참 좋네요.\n"
        "#Person2#: 네, 정말이에요. 이런 날은 어디라도 놀러 가고 싶어요."
    )

    interface = gr.Interface(
        fn=summarize,
        inputs=gr.Textbox(lines=15, label="Dialogue", placeholder="Enter dialogue here..."),
        outputs=gr.Textbox(lines=5, label="Summary", interactive=True),
        title="Korean Dialogue Summarization Demo",
        description="Enter a Korean dialogue to summarize it. This is a basic template without a model loaded.",
        examples=[[example_dialogue]],
        allow_flagging='never'
    )
    
    return interface

if __name__ == "__main__":
    print("Starting Gradio App...")
    
    # To run this script, use the command:
    # python src/app/app_gradio.py
    
    app = create_gradio_app()
    app.launch()
