from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.tools import tool

model1 = ChatOpenAI(model="gpt-4o")
model2 = ChatOpenAI(model="gpt-4o")
model3 = ChatOpenAI(model="gpt-4o")


@tool
def arithmetic_tool(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation (str): One of '+', '-', '*', '/', '**', '%'
        a (float): First operand
        b (float): Second operand

    Returns:
        float: Result of the operation

    Raises:
        ValueError: If operation is unsupported or division by zero
    """
    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ValueError("Division by zero")),
        '**': lambda x, y: x ** y,
        '%': lambda x, y: x % y if y != 0 else (_ for _ in ()).throw(ValueError("Modulo by zero")),
    }

    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")

    return operations[operation](a, b)

@tool
def welcome_messgage()-> str:
    """ Handles greeting message """
    return " Hi , How can i help you today"

welcome_tools = [welcome_messgage, create_handoff_tool(agent_name="calculator_agent"), create_handoff_tool(agent_name="weather_agent")]
welcome_agent = create_react_agent(
    model1.bind_tools(welcome_tools, parallel_tool_calls=False), 
    welcome_tools,
    prompt="You are a welcome agent, Whenever a greetings says, you need to do th greeting back properly",
    name="welcome_agent"
)

calculator_tools = [arithmetic_tool, create_handoff_tool(agent_name="weather_agent"),  create_handoff_tool(agent_name="welcome_agent")]
calculator_agent = create_react_agent(
    model2.bind_tools(calculator_tools, parallel_tool_calls=False), 
    calculator_tools,
    prompt="Calculator agent for add, subtract , multiplication and division",
    name="calculator_agent"
)

@tool
def get_weather(place: str) -> int:
    """Return the current temperature in Fahrenheit for a given place."""
    place = place.lower().strip()  # Normalize input
    if place == "newyork":
        return 55  # ~13°C — typical spring/fall day
    elif place == "chicago":
        return 30  # ~ -1°C — plausible winter day
    else:
        return 40  # Default fallback (~4°C)

weather_tools = [get_weather, create_handoff_tool(agent_name="calculator_agent", description="This agent is responsible to calculate arithemtic operations"), 
                create_handoff_tool(agent_name="welcome_agent", description="This agnet is responsible for the welcome messages, Any user greetings will be handled by this agent")]
weather_agent = create_react_agent(
    model3.bind_tools(weather_tools, parallel_tool_calls=False), 
    weather_tools,
    prompt="An agent to find the weather in a place",
    name="weather_agent"
)

checkpointer = InMemorySaver()

workflow = create_swarm(
    [calculator_agent, weather_agent, welcome_agent],
    default_active_agent="welcome_agent"
)

app_langraph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "11"}}



app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple mock responses for demo - replace with your actual AI/chat logic
SAMPLE_RESPONSES = [
    "Hello! I'm here to help you with any questions you might have.",
    "That's an interesting question. Let me think about it for a moment.",
    "Based on what you're asking, I would suggest considering multiple perspectives.",
    "Here's what I know about that topic and how it might be relevant to your situation.",
    "I hope this information helps clarify things for you. Feel free to ask follow-up questions."
]



async def generate_word_stream(message: str):

    async for event in app_langraph.astream_events(
    {
        "messages": [
            {"role": "user", "content": message}
        ]
       },
        config,
    ):
        if event.get("event") == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk")

            if not chunk:
                continue

            content = getattr(chunk, "content", "")
            additional_kwargs = getattr(chunk, "additional_kwargs", {})
            response_metadata = getattr(chunk, "response_metadata", {})

            if content and additional_kwargs == {}:
                print(content, end="", flush=True)
                await asyncio.sleep(0.01)  # ✅ async-friendly sleep
                r1 = f"data: {json.dumps({'word': content, 'done': False})}\n\n"
                yield r1

            if response_metadata.get("finish_reason") == "stop":
                yield f"data: {json.dumps({'word': '', 'done': True})}\n\n"
                break



@app.post("/chat/stream")
async def stream_chat_response(request: dict):
    """Stream chat response word by word"""
    message = request.get("message", "")
    
    if not message:
        return {"error": "Message is required"}
    
    return StreamingResponse(
        generate_word_stream(message),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )