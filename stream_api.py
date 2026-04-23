from flask import Response, request
from llm_chain import chains
import json

# This route streams the model response token by token

def stream_response():
    user_input = request.json.get("message")
    result = chains.invoke({'topic': user_input}, config={"timeout": 10})
    explanation = result.get("explain", "")
    summary = result.get("summary", "")
    example = result.get("example", "")
    links = result.get("links", "")
    def generate():
        yield '{"explain": "'
        for char in explanation:
            yield char
        yield '", "summary": "'
        for char in summary:
            yield char
        yield '", "example": "'
        for char in example:
            yield char
        yield '", "links": "'
        for char in links:
            yield char
        yield '"}'
    return Response(generate(), mimetype='application/json')
