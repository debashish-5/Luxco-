from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
from duckduckgo_search import DDGS

from langchain.tools import tool

@tool
def farming_links(topic:str) -> str:
    """
    Return real-time farming-related URLs for a given query.
    No API key required
    """
    search_query = f"{topic} farming agriculture"
    try:
        result = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query,max_results=5):
                title = r.get("title","No title")
                url = r.get("href","")
                body = r.get("body","")
                if url:
                    result.append(
                        f"{title}\n{url}\n{body}\n"
                    )
        if not result:
            return "No links found for this topic."
        return "\n".join(result)
    except Exception as e:
        return f"Search error:{e}"
    
# Model
model = ChatOllama(model="mistral")

# Prompts
explain_prompt = PromptTemplate.from_template("Explain this {topic}")
summary_prompt = PromptTemplate.from_template("Summarize this {topic}")
example_prompt = PromptTemplate.from_template("Give 3 examples of this {topic}")

predict_prompt = PromptTemplate.from_template("""
User input:
Crop:{crop}
Problem:{problem}
Weather:{weather}
Soil:{soil}
Stage:{stage}
Severity:{severity}
Model Prediction: {prediction}

You are an AI Agricultural Expert:
- Evaluate the prediction
- Suggest improvements
- Explain clearly
""")


tool_prompt = PromptTemplate.from_template("""
You are an AI Agriculture Assistant.

User query: {query}

If the user is asking for resources, guides, or help related to farming,
you MUST use the tool `farming_links`.

Otherwise, answer normally.
""")


parser = StrOutputParser()

# Chains
explain_chain = explain_prompt | model | parser
summary_chain = summary_prompt | model | parser
example_chain = example_prompt | model | parser
predict_chain = predict_prompt | model | parser
tool_chain = tool_prompt | model | parser

chains = RunnableParallel(
    explain=explain_chain,
    summary=summary_chain,
    example=example_chain,
    links=RunnableLambda(lambda x:farming_links.invoke(x["topic"]))
)
  