import gradio as gr
import spaces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Question: {Question}")
])


@spaces.GPU
def generate_response(query, api_key, llm, temperature, max_tokens):
    if not query:
        return "Please enter a question."

    if not api_key:
        return "Please enter your OpenAI API key."

    try:
        model = ChatOpenAI(
            model=llm,
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens
        )

        parser = StrOutputParser()
        chain = prompt | model | parser

        result = chain.invoke({
            "Question": query
        })

        return result

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="QnA Chatbot") as demo:

    gr.Markdown(
        """
        # 🤖 QnA Chatbot
        Ask a question and get an answer from OpenAI.
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            query = gr.Textbox(
                label="Your Question",
                placeholder="What would you like to ask?",
                lines=4
            )

            answer = gr.Textbox(
                label="Answer",
                lines=10
            )

            ask_button = gr.Button(
                "🚀 Ask",
                variant="primary"
            )

        with gr.Column(scale=1):

            gr.Markdown("### ⚙️ Settings")

            api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API key",
                type="password"
            )

            llm = gr.Dropdown(
                choices=[
                    "gpt-4.1-nano",
                    "gpt-3.5-turbo",
                    "gpt-4.1",
                    "gpt-4.1-mini"
                ],
                value="gpt-4.1-nano",
                label="Select LLM"
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Temperature"
            )

            max_tokens = gr.Slider(
                minimum=50,
                maximum=1000,
                value=200,
                step=50,
                label="Max Tokens"
            )

    ask_button.click(
        fn=generate_response,
        inputs=[
            query,
            api_key,
            llm,
            temperature,
            max_tokens
        ],
        outputs=answer
    )


if __name__ == "__main__":
    demo.launch()
