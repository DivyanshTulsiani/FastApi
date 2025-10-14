from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import re, json
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


class DiagramRequest(BaseModel):
    user_id: str
    prompt: str
    
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

@app.get("/")
async def read_root():
 return {"message": "Hello World"}



@app.post("/generate/diagram")
async def generate_diagram(req: DiagramRequest):
    # vectorstore_path = user_stores.get(req.user_id, None)
    
    # PDF is only extra context
    context = ""

    # if vectorstore_path and os.path.exists(vectorstore_path):
    #     # Create a fresh ChromaDB connection each time
    #     vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    #     docs = vectorstore.similarity_search(req.prompt, k=3)
    #     context = "\n".join([d.page_content for d in docs])

    template = """
  You are a diagram generator.
Your task is to create flowcharts, system designs, or process diagrams based on the user request. 
Use the extra context if provided.

User request: {user_prompt}

Extra context (optional, may be empty): {context}

Rules:
- Always generate a diagram, even if no extra context is given.
- Return ONLY valid JSON (no text outside JSON).
- JSON must have this exact structure for React Flow:
- Don't add colors or shape in nodes data until specified
- Use light and subtle colors to make it look elegant

{{
  "nodes": [
    {{
      "id": "unique-string",
      "type": "input" | "default" | "output",
      {{
        "label": "string",
        "color": "string (optional)",
        "shape": "string (optional)"
      }},
      "position": {{ "x": number, "y": number }}
    }}
  ],
  "edges": [
    {{
      "id": "unique-string",
      "source": "id-of-source-node",
      "target": "id-of-target-node"
    }}
  ]
}}

Additional guidelines:
- Use "input" type for starting nodes, "output" type for ending nodes, "default" for everything else.
- Position can be a placeholder (e.g., {{ "x": 0, "y": 0 }}) if layout is not known.
- Ensure every edge connects existing nodes.
- Ensure unique IDs for all nodes and edges.
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain = prompt | llm 

    try:
        result = await chain.ainvoke({
            "user_prompt": req.prompt,
            "context": context
        })

        raw_text = result.content.strip()
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", raw_text.strip())

        try:
            data = json.loads(cleaned)
            # if collection_name:
            #   # Delete the collection, which safely releases the lock
            #   persistent_client.delete_collection(collection_name)
            #   # Remove the mapping from the in-memory store
            #   user_collections.pop(req.user_id, None)
            return {
                "parsed": data,
                "retrieved_context": context
                  }

        except Exception as e:
            return {"raw": raw_text, "error": f"Still invalid JSON: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}