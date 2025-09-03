import gradio as gr
from eis_rag.rag_chain import build_chain

run = build_chain()

def respond(user_msg, history):
    answer, _ = run(user_msg)
    return history + [(user_msg, answer)], ""

with gr.Blocks(title="Engineering Innovation Studio Assistant") as demo:
    gr.Markdown("# 🛠️ Engineering Innovation Studio Assistant")
    chat = gr.Chatbot(height=480)
    msg = gr.Textbox(placeholder="Ask about services, 3D printing, electronics, hours...")
    gr.ClearButton([chat])
    msg.submit(respond, [msg, chat], [chat, msg])

if __name__ == "__main__":
    demo.launch()