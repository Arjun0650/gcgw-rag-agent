
import os
import re
import json

from pathlib import Path
from typing import TypedDict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langgraph.graph import StateGraph, START, END


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# OWNER CONTACT DETAILS
# =========================================================

OWNER_PHONE = os.getenv(
    "OWNER_PHONE",
    "YOUR_OWNER_PHONE"
)

OWNER_EMAIL = os.getenv(
    "OWNER_EMAIL",
    "YOUR_OWNER_EMAIL@gmail.com"
)


# =========================================================
# RAG MATCH THRESHOLD
# =========================================================
# Higher value = stricter matching.
# Unknown questions will go to owner contact.

MATCH_THRESHOLD = float(
    os.getenv("MATCH_THRESHOLD", "0.78")
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

def load_knowledge():

    path = BASE_DIR / "knowledge_base.json"

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


KNOWLEDGE = load_knowledge()


# =========================================================
# CONVERT KNOWLEDGE TO LANGCHAIN DOCUMENTS
# =========================================================

documents = []

for item in KNOWLEDGE:

    document = Document(

        page_content=f"""
Question:
{item['question']}

Answer:
{item['answer']}
""".strip(),

        metadata={
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"]
        }

    )

    documents.append(document)


# =========================================================
# CREATE CHROMADB VECTOR STORE
# =========================================================

vector_store = Chroma.from_documents(

    documents=documents,

    embedding=embeddings,

    collection_name="gcgw_faq",

    collection_metadata={
        "hnsw:space": "cosine"
    }

)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(
    TypedDict,
    total=False
):

    question: str

    retrieved: List[Document]

    score: float

    answer: str

    route: str


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize(text):

    text = text.strip().lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =========================================================
# HELPER: CHECK KEYWORDS
# =========================================================

def contains_any(
    question,
    keywords
):

    return any(
        keyword in question
        for keyword in keywords
    )


# =========================================================
# RAG RETRIEVAL NODE
# =========================================================

def retrieve_node(
    state: AgentState
):

    question = state["question"].strip()

    normalized_question = normalize(
        question
    )


    # -----------------------------------------------------
    # STEP 1: EXACT QUESTION MATCH
    # -----------------------------------------------------

    for item in KNOWLEDGE:

        stored_question = normalize(
            item["question"]
        )

        if normalized_question == stored_question:

            doc = Document(

                page_content=item["answer"],

                metadata=item

            )

            return {

                "retrieved": [doc],

                "score": 1.0,

                "route": "known"

            }


    # -----------------------------------------------------
    # STEP 2: DEFINE TOPICS
    # -----------------------------------------------------

    topic_keywords = {

        "gcgw_intro": [

            "what is gcgw",

            "who is gcgw",

            "tell me about gcgw",

            "about gcgw",

            "about company",

            "about your company",

            "company details",

            "company information",

            "who are you",

            "what is your company"

        ],


        "gcgw_projects": [

            "how many projects",

            "projects completed",

            "completed projects",

            "project completed",

            "project count",

            "project locations",

            "where have you worked",

            "where are your projects",

            "projects till now",

            "projects this year"

        ],


        "gcgw_services": [

            "services",

            "service",

            "what services",

            "what service",

            "what work do you do",

            "what works do you do",

            "what do you do",

            "guniting",

            "structural repair",

            "structural repairs",

            "concrete rehabilitation",

            "civil engineering",

            "structural strengthening",

            "repair work"

        ],


        "gcgw_start": [

            "when started",

            "when did gcgw start",

            "starting year",

            "start year",

            "started year",

            "founded",

            "established",

            "when established",

            "when founded",

            "since when",

            "company started",

            "gcgw started"

        ]

    }


    # -----------------------------------------------------
    # STEP 3: MANUAL TOPIC MATCHING
    # -----------------------------------------------------

    detected_topic = None


    for topic_id, keywords in topic_keywords.items():

        if contains_any(
            normalized_question,
            keywords
        ):

            detected_topic = topic_id

            break


    # -----------------------------------------------------
    # STEP 4: IF TOPIC FOUND, RETURN CORRECT ANSWER
    # -----------------------------------------------------

    if detected_topic:

        for item in KNOWLEDGE:

            if item["id"] == detected_topic:

                doc = Document(

                    page_content=item["answer"],

                    metadata=item

                )

                return {

                    "retrieved": [doc],

                    "score": 1.0,

                    "route": "known"

                }


    # -----------------------------------------------------
    # STEP 5: SEMANTIC RAG SEARCH
    # -----------------------------------------------------

    results = (
        vector_store
        .similarity_search_with_relevance_scores(
            question,
            k=1
        )
    )


    if not results:

        return {

            "retrieved": [],

            "score": 0.0,

            "route": "unknown"

        }


    document, score = results[0]

    score = float(score)


    # -----------------------------------------------------
    # STEP 6: CHECK RETRIEVED TOPIC
    # -----------------------------------------------------

    retrieved_id = document.metadata.get(
        "id",
        ""
    )


    retrieved_keywords = topic_keywords.get(
        retrieved_id,
        []
    )


    keyword_match = contains_any(
        normalized_question,
        retrieved_keywords
    )


    # -----------------------------------------------------
    # STEP 7: STRICT FINAL VALIDATION
    # -----------------------------------------------------

    if (
        score >= MATCH_THRESHOLD
        and keyword_match
    ):

        route = "known"

    else:

        route = "unknown"


    return {

        "retrieved": [document],

        "score": score,

        "route": route

    }


# =========================================================
# KNOWN ANSWER NODE
# =========================================================

def known_answer_node(
    state: AgentState
):

    document = state["retrieved"][0]


    answer = document.metadata.get(

        "answer",

        document.page_content

    )


    return {

        "answer": answer

    }


# =========================================================
# UNKNOWN QUESTION NODE
# =========================================================

def unknown_answer_node(
    state: AgentState
):

    answer = f"""
I don't have a confirmed answer for that in the GCGW knowledge base.

For more information, please contact the owner directly.

Phone: {OWNER_PHONE}

Email: {OWNER_EMAIL}
""".strip()


    return {

        "answer": answer

    }


# =========================================================
# LANGGRAPH ROUTER
# =========================================================

def route_question(
    state: AgentState
):

    return state.get(
        "route",
        "unknown"
    )


# =========================================================
# BUILD LANGGRAPH AGENT
# =========================================================

builder = StateGraph(
    AgentState
)


builder.add_node(

    "retrieve",

    retrieve_node

)


builder.add_node(

    "known_answer",

    known_answer_node

)


builder.add_node(

    "unknown_answer",

    unknown_answer_node

)


builder.add_edge(

    START,

    "retrieve"

)


builder.add_conditional_edges(

    "retrieve",

    route_question,

    {

        "known":
        "known_answer",

        "unknown":
        "unknown_answer"

    }

)


builder.add_edge(

    "known_answer",

    END

)


builder.add_edge(

    "unknown_answer",

    END

)


agent = builder.compile()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(

    title="GCGW RAG Agent",

    version="2.0.0",

    description="""
GCGW website chatbot using
RAG + LangChain + ChromaDB + LangGraph
"""

)


# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "https://gcgunitingwork.netlify.app",

        "http://localhost:3000",

        "http://127.0.0.1:5500"

    ],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]

)


# =========================================================
# API REQUEST MODEL
# =========================================================

class ChatRequest(
    BaseModel
):

    message: str


# =========================================================
# API RESPONSE MODEL
# =========================================================

class ChatResponse(
    BaseModel
):

    answer: str

    matched: bool

    confidence: float

    route: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get(
    "/health"
)

def health():

    return {

        "status":
        "healthy",

        "service":
        "GCGW RAG Agent",

        "version":
        "2.0",

        "knowledge_items":
        len(KNOWLEDGE)

    }


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)

def chat(
    req: ChatRequest
):

    message = req.message.strip()


    if not message:

        return ChatResponse(

            answer="Please type a question.",

            matched=False,

            confidence=0.0,

            route="unknown"

        )


    # Run LangGraph Agent

    result = agent.invoke({

        "question":
        message

    })


    route = result.get(
        "route",
        "unknown"
    )


    matched = (
        route == "known"
    )


    confidence = round(

        float(

            result.get(
                "score",
                0.0
            )

        ),

        3

    )


    return ChatResponse(

        answer=result.get(

            "answer",

            "Unable to answer."

        ),

        matched=matched,

        confidence=confidence,

        route=route

    )


# =========================================================
# BASIC CHAT UI
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)

def home():

    return """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
GCGW AI Assistant
</title>


<style>


* {

box-sizing:
border-box;

}


body {

margin:
0;

min-height:
100vh;

display:
flex;

justify-content:
center;

align-items:
center;

background:
#071b36;

font-family:
Arial,
Helvetica,
sans-serif;

padding:
20px;

}


.chat-container {

width:
700px;

max-width:
100%;

background:
white;

border-radius:
20px;

overflow:
hidden;

box-shadow:
0 20px 60px
rgba(0,0,0,.35);

}


.chat-header {

background:
#0b274b;

color:
white;

padding:
24px;

}


.chat-header h1 {

margin:
0;

font-size:
24px;

}


.chat-header p {

margin:
8px 0 0;

opacity:
.8;

}


.messages {

height:
450px;

overflow-y:
auto;

padding:
20px;

background:
#f4f7fb;

}


.message {

max-width:
82%;

padding:
13px 16px;

margin:
10px 0;

border-radius:
15px;

line-height:
1.5;

white-space:
pre-wrap;

}


.bot {

background:
white;

border:
1px solid #dce3eb;

}


.user {

background:
#0b274b;

color:
white;

margin-left:
auto;

}


.controls {

display:
flex;

gap:
10px;

padding:
15px;

border-top:
1px solid #eee;

}


input {

flex:
1;

padding:
14px;

font-size:
15px;

border:
1px solid #ccd3dd;

border-radius:
10px;

outline:
none;

}


button {

border:
0;

background:
#0b274b;

color:
white;

padding:
0 22px;

border-radius:
10px;

font-weight:
bold;

cursor:
pointer;

}


button:hover {

background:
#123b70;

}


</style>


</head>


<body>


<div class="chat-container">


<div class="chat-header">


<h1>
GCGW AI Assistant
</h1>


<p>
Gopal Chavan Guniting Work
</p>


</div>


<div
class="messages"
id="messages">


<div class="message bot">

Hello! 👋

Welcome to GCGW.

You can ask me about:

• GCGW
• Our services
• Completed projects
• Company starting year

For other questions, I will provide the owner's contact information.

</div>


</div>


<div class="controls">


<input

id="question"

type="text"

placeholder="Ask something about GCGW..."

autocomplete="off"

>


<button
onclick="sendMessage()">

Send

</button>


</div>


</div>


<script>


const input =
document.getElementById(
"question"
);


const messages =
document.getElementById(
"messages"
);


function addMessage(
text,
type
) {

const div =
document.createElement(
"div"
);

div.className =
"message " + type;

div.textContent =
text;

messages.appendChild(
div
);

messages.scrollTop =
messages.scrollHeight;

}


async function sendMessage() {


const question =
input.value.trim();


if (!question) {

return;

}


addMessage(
question,
"user"
);


input.value =
"";


try {


const response =
await fetch(

"/chat",

{

method:
"POST",

headers: {

"Content-Type":
"application/json"

},

body:
JSON.stringify({

message:
question

})

}

);


const data =
await response.json();


addMessage(

data.answer,

"bot"

);


}


catch (error) {


addMessage(

"The GCGW assistant is temporarily unavailable. Please try again.",

"bot"

);


}


}


input.addEventListener(

"keydown",

function(event) {


if (
event.key === "Enter"
) {

sendMessage();

}


}

);


</script>


</body>

</html>

"""


# =========================================================
# LOCAL / RENDER SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host=
        "0.0.0.0",

        port=
        int(

            os.getenv(
                "PORT",
                "8000"
            )

        )

    )
