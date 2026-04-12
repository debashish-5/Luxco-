from flask import Flask, render_template, request, jsonify
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOllama(model="mistral")

explain_prompt = PromptTemplate.from_template("Explain this {topic}")
summary_prompt = PromptTemplate.from_template("Summarize this {topic}")
example_prompt = PromptTemplate.from_template("Give 3 example of this {topic}")

parser = StrOutputParser()

explain_chain = explain_prompt | model | parser
summary_chain = summary_prompt | model | parser
example_chain = example_prompt | model | parser

chains = RunnableParallel(
    explain=explain_chain,
    summary=summary_chain,
    example=example_chain
)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get",methods=['POST'])
def get_bot_response():
    user_input = request.json.get("message")
    result = chains.invoke({'topic': user_input}, config={"timeout": 10})
    return jsonify({"reply":result})
if __name__ == "__main__":
    app.run(debug=True)