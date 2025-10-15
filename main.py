from fastapi import FastAPI, UploadFile, File
import uvicorn
import chromadb
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import re, json
import os
import tempfile
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


user_collections = {}

persistent_client = chromadb.EphemeralClient()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = "jina-embeddings-v3"
JINA_DIM = 512


class DiagramRequest(BaseModel):
    user_id: str
    prompt: str
    

# --- Jina embedding function (robust to Jina API return shapes) ---
def jina_embed(texts):
    """
    texts: list[str]
    Returns: list[list[float]] (one embedding per text)
    """
    if isinstance(texts, str):
        texts = [texts]

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": JINA_MODEL,
        "task": "text-matching",
        "input": texts  # list of strings (this matches the working REST usage you showed)
    }

    resp = requests.post(url, headers=headers, json=payload)
    try:
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        # helpful debug print for local dev (remove or reduce in production)
        print("Jina API request failed:", resp.status_code, resp.text)
        raise

    # Support both response shapes we've seen in various docs/wrappers:
    # - {"data":[{"index":0,"embedding":[...]} , ...]}
    # - {"embeddings":[ [...], [...], ... ]}
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], list):
            return [item["embedding"] for item in result["data"]]
        if "embeddings" in result:
            return result["embeddings"]
    raise ValueError("Unexpected Jina response shape: " + str(result))


# --- Wrapper that matches Chroma's EmbeddingFunction interface strictly ---
class JinaEmbeddingWrapper:
    def __init__(self, embed_fn, dimension):
        # embed_fn must accept list[str] and return list[list[float]]
        self._embed_fn = embed_fn
        self._dim = dimension

    # REQUIRED: __call__(self, input)
    def __call__(self, input):
        # Chromadb will pass either a list[str] or a single str.
        if isinstance(input, str):
            input_list = [input]
            embeddings = self._embed_fn(input_list)
            # return list for consistency with __call__
            return embeddings
        # assume it's already an iterable of strings
        return self._embed_fn(list(input))

    # Chromadb may call embed_documents(input=...) when adding vectors
    def embed_documents(self, input):
        # signature must be (input) - no **kwargs
        if isinstance(input, str):
            input_list = [input]
            return self._embed_fn(input_list)
        return self._embed_fn(list(input))

    # Chromadb may call embed_query(input=...) when querying
    def embed_query(self, input):
    # Accept either a single string or a list with one string.
        if isinstance(input, str):
          input_list = [input]
        else:
          input_list = list(input)

        embs = self._embed_fn(input_list)

    # Chroma expects list[list[float]]
        return embs
        # if the list length is 1, return the single vector (chroma sometimes expects single)

    # Chromadb expects a name() method in some validation paths
    def name(self):
        return "jina-embedding"

    @property
    def embedding_dimension(self):
        return self._dim

@app.get("/")
async def read_root():
 return {"message": "Hello World"}

@app.post("/upload-pdf/{user_id}")
async def upload_pdf(user_id: str, file: UploadFile = File(...)):
    #we have to sanitize user id as collection name should not include special characters
    sanitized_user_id = re.sub(r'[^a-zA-Z0-9._-]', '_', user_id)
    collection_name = f"user-{sanitized_user_id}-docs"
    persistent_client.list_collections()
    # this checks if connection present before
    try:
        persistent_client.delete_collection(collection_name)
    except Exception as e:
        # This will raise an exception if the collection doesn't exist,
        # which we can safely ignore.
        print(f"Collection '{collection_name}' not found, creating a new one.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # currently using PyPDFLoader will shift if available
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)


    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    texts = [doc.page_content[:8000] for doc in docs] 
    
    #new way we are creating a chromadb client instead of storing it directly in disk
     # The `get_or_create_collection` method is crucial.
    # It gets the collection if it exists or creates a new one,
    # ensuring no overwrites or file-locking conflicts.
    jina_embedding = JinaEmbeddingWrapper(jina_embed, JINA_DIM)

    collection = persistent_client.get_or_create_collection(
      name=collection_name, embedding_function=jina_embedding
    )
    # #this deletes this users old items
    # collection.delete(where={})

        # 5. Add documents to the collection
    # The native `add` method takes content, metadata, and unique IDs

    try:
        vectors = jina_embed(texts)
    except Exception as e:
        print("Jina API failed:", e)
        return {"error": f"Jina API failed: {str(e)}"}

    collection.add(
      documents=[doc.page_content for doc in docs],
      metadatas=[doc.metadata for doc in docs],
      ids=[f"id-{sanitized_user_id}-{i}" for i in range(len(docs))],
      embeddings=vectors  # directly pass embeddings to avoid multiple calls
    )

    # vectorstore = FAISS.from_documents(docs, embeddings)
    # print("Vectorstore built for user:", user_id, "with", len(docs), "chunks")

#     query = "leetcode"  # replace with something from your PDF
#     results = db.similarity_search(query, k=2)  # top 2 similar chunks
#     print(f"Similarity search for query: '{query}'")
#     for i, r in enumerate(results):
#       print(f"Result {i}:", r.page_content[:300]) 
# # weaviate
#     user_stores[user_id] = vectorstore_path

    user_collections[user_id] = collection_name
    return {"status": "PDF uploaded and processed", "user_id": user_id}



@app.post("/generate/diagram")
async def generate_diagram(req: DiagramRequest):
    # vectorstore_path = user_stores.get(req.user_id, None)
    
    collection_name = user_collections.get(req.user_id, None)
    
    # PDF is only extra context
    context = ""
    jina_embedding = JinaEmbeddingWrapper(jina_embed, JINA_DIM)
    
    if collection_name:
        try:
            # 1. Get the user's collection from the persistent client
            collection = persistent_client.get_collection(
                name=collection_name, embedding_function=jina_embedding
            )
            
            # 2. Query the collection using the user's prompt
            results = collection.query(
                query_texts=[req.prompt],
                n_results=3
            )
            # The result is a dictionary, extract the documents
            context = "\n".join(results['documents'][0])
        except Exception as e:
            print(f"Error retrieving context for user {req.user_id}: {e}")

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
      
      

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)