import os
import numpy as np
import faiss
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import json
from datetime import datetime

# --- 1. Streamlit and Web Recorder ---
import streamlit as st
from st_audiorec import st_audiorec

# --- 2. Voice API Libraries ---
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud.texttospeech import SsmlVoiceGender, AudioEncoding, SynthesisInput, VoiceSelectionParams, AudioConfig

# --- 3. Audio processing libraries ---
import io
from pydub import AudioSegment

# --- 0. Initialize GCP and models ---
PROJECT_ID = "wis-exercise-4-api"  
LOCATION = "europe-west1"            
@st.cache_resource
def load_models_and_data():
    print("Loading all models and data (only once)...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # --- Load models ---
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    llm = GenerativeModel("gemini-2.5-flash")
    
    # --- [!!] 1. CRITICAL FIX: Define 'dtypes' ---
    # Force all ID columns to be read as 'str' (string/text).
    customer_dtype = {'customer_id': str}
    product_dtype = {'product_id': str, 'customer_id': str}
    transaction_dtype = {'product_id': str} # transaction_id is not relevant

    print("  -> Loading all CSV databases...")

    # --- [!!] Load all 4 CSV files ---
    try:
        customers_df = pd.read_csv("./synthetic_data/customers.csv", dtype=customer_dtype)
        products_df = pd.read_csv("./synthetic_data/products.csv", dtype=product_dtype)
        products_closed_df = pd.read_csv("./synthetic_data/products_closed.csv", dtype=product_dtype)
        transactions_df = pd.read_csv("./synthetic_data/transactions.csv", dtype=transaction_dtype)
        
        # [!!] Important: convert date columns for comparisons
        transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
        
        print("All CSV data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}.")
        st.error(f"Error: Data file not found - {e}. Please ensure the './synthetic_data/' directory exists.")
        st.stop()
        
    # --- [!!] Register all tools with the LLM ---
    # We will define 'tools' after the function declarations
    tool_llm = GenerativeModel("gemini-2.5-flash", tools=all_tools)

    # --- Load RAG index ---
    index = faiss.read_index("ing_chunks_nl.index")
    chunk_file_map = []
    with open("ing_chunks_nl.map", 'r', encoding='utf-8') as f:
        for line in f:
            chunk_file_map.append(line.strip())
            
    print("...models and data loaded.")
    # return all needed resources
    return (embedding_model, llm, tool_llm, index, chunk_file_map, 
            customers_df, products_df, products_closed_df, transactions_df)

# --- Simulate the currently logged-in user ---
# --- [!!] 2. CRITICAL FIX: Treat the ID as a string ---
CURRENT_CUSTOMER_ID: str = "1001" # Must be a string to match CSV dtype

# ===============================================
# --- [!!] NEW: CSV data tools (Python functions) ---
# ===============================================
# These functions will access the global DFs returned by load_models_and_data

# [!!] 3. CRITICAL FIX: Use 'str' type-hinting for all IDs
def get_customer_details(customer_id: str):
    """Retrieve customer details (address, phone, email) by customer ID."""
    details = customers_df[customers_df['customer_id'] == customer_id]
    if details.empty:
        return json.dumps({"error": "Customer not found"})
    return details.to_json(orient='records')

def get_customer_products(customer_id: str, include_closed: bool = False):
    """Retrieve all products for a customer by customer ID. By default only active/frozen products are shown."""
    products = products_df[products_df['customer_id'] == customer_id]
    if include_closed:
        closed_products = products_closed_df[products_closed_df['customer_id'] == customer_id]
        products = pd.concat([products, closed_products])
    return products.to_json(orient='records')

def get_account_balance(product_id: str):
    """Compute the current balance for a specific product ID (e.g., current or savings account) by aggregating transactions."""
    product_transactions = transactions_df[transactions_df['product_id'] == product_id]
    if product_transactions.empty:
        return json.dumps({"product_id": product_id, "balance": 0, "currency": "EUR"})
        
    total_credit = product_transactions[product_transactions['transaction_type'] == 'Credit']['amount'].sum()
    total_debit = product_transactions[product_transactions['transaction_type'] == 'Debit']['amount'].sum()
    balance = total_credit - total_debit
    
    return json.dumps({"product_id": product_id, "balance": f"{balance:.2f}", "currency": "EUR"})

def get_recent_transactions(product_id: str, limit: int = 5):
    """Retrieve the most recent 'limit' transactions for a specific product ID (default is 5)."""
    product_transactions = transactions_df[transactions_df['product_id'] == product_id]
    # Sort by date in descending order and get the top 'limit' transactions
    recent = product_transactions.sort_values(by='date', ascending=False).head(limit)
    return recent.to_json(orient='records')

def get_transactions_by_date_range(product_id: str, start_date: str, end_date: str):
    """Retrieve all transactions for a specific product ID within a date range (date format YYYY-MM-DD)."""
    # Convert input string dates to datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # The 'product_id' is now a string (e.g., "2001")
    product_transactions = transactions_df[transactions_df['product_id'] == product_id]
    # Filter by date range (we previously converted the 'date' column to datetime)
    mask = (product_transactions['date'] >= start) & (product_transactions['date'] <= end)
    transactions_in_range = product_transactions[mask]
    
    return transactions_in_range.to_json(orient='records')

# ===============================================
# --- [!!] New: Tool Declarations ---
# ===============================================
# (Here we adjust the 'type' to 'string' for clarity)

get_customer_details_declaration = FunctionDeclaration(
    name="get_customer_details",
    description="Retrieves customer details (address, phone) for a customer ID.",
    parameters={
        "type": "object", "properties": {"customer_id": {"type": "string", "description": "The ID of the customer (e.g., '1001')"}}, "required": ["customer_id"]
    }
)
get_customer_products_declaration = FunctionDeclaration(
    name="get_customer_products",
    description="Retrieves a list of all products (accounts, cards) for a customer.",
    parameters={
        "type": "object", 
        "properties": {
            "customer_id": {"type": "string", "description": "The ID of the customer (e.g., '1001')"},
            "include_closed": {"type": "boolean", "description": "Include closed products (default false)"}
        }, 
        "required": ["customer_id"]
    }
)
get_account_balance_declaration = FunctionDeclaration(
    name="get_account_balance",
    description="Calculates the current balance for a specific product ID (account).",
    parameters={
        "type": "object", "properties": {"product_id": {"type": "string", "description": "The ID of the product (e.g., '2001')"}}, "required": ["product_id"]
    }
)
get_recent_transactions_declaration = FunctionDeclaration(
    name="get_recent_transactions",
    description="Retrieves the most recent transactions for a specific product ID.",
    parameters={
        "type": "object", 
        "properties": {
            "product_id": {"type": "string", "description": "The ID of the product (e.g., '2001')"},
            "limit": {"type": "integer", "description": "Number of transactions (default 5)"}
        }, 
        "required": ["product_id"]
    }
)
get_transactions_by_date_range_declaration = FunctionDeclaration(
    name="get_transactions_by_date_range",
    description="Retrieves all transactions for a product ID between two dates.",
    parameters={
        "type": "object", 
        "properties": {
            "product_id": {"type": "string", "description": "The ID of the product (e.g., '2001')"},
            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
        }, 
        "required": ["product_id", "start_date", "end_date"]
    }
)

# --- List of all tools ---
all_tools = [
    Tool(function_declarations=[
        get_customer_details_declaration,
        get_customer_products_declaration,
        get_account_balance_declaration,
        get_recent_transactions_declaration,
        get_transactions_by_date_range_declaration
    ]),
]

# --- Load all resources ---
(embedding_model, llm, tool_llm, index, chunk_file_map, 
 customers_df, products_df, products_closed_df, transactions_df) = load_models_and_data()

CHUNKS_DIR_NL = "./chunks/500_750_processed_be_nl_2025_09_23/"


# ===============================================
# --- 3. Core "Brain" (RAG + Function Calling) ---
# ===============================================

def answer_with_rag(user_query: str) -> str:
    """Answer general questions using RAG (vector search)."""
    print("  -> Answering general question using RAG process.")
    query_embedding = embedding_model.get_embeddings([user_query])[0].values
    query_vector_np = np.array([query_embedding], dtype='float32')
    k = 3
    distances, indices = index.search(query_vector_np, k)
    context_chunks = []
    for i, idx in enumerate(indices[0]):
        filename = chunk_file_map[idx]
        with open(os.path.join(CHUNKS_DIR_NL, filename), 'r', encoding='utf-8') as f:
            context_chunks.append(f.read())
    context_for_llm = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
    [System Instruction]
    You are a helpful ING bank assistant.
    Please answer the user's question in Dutch based solely on the provided "context" below.
    If there is not enough information in the context, politely respond that you don't know and do not make up an answer.
    [Context]
    {context_for_llm}
    [User Question]
    {user_query}
    [Your Answer (in Dutch only)]
    """
    print("  -> Will answer using RAG...")
    response = llm.generate_content(prompt)
    return response.text.strip()

# --- [!!] Update: Execute all new tools ---
# [!!] 4. CRITICAL FIX: Remove 'int()' conversion. We now work with strings.
def execute_tool_call(tool_call):
    """(Brain-Branch 2) Tool Executor: Executes Python functions."""
    tool_name = tool_call.name
    tool_args = tool_call.args
    print(f"  -> [Brain] Tool called: {tool_name} (Arguments: {tool_args})")

    # 1. Tools that do *not* require product_id
    if tool_name == "get_customer_details":
        result = get_customer_details(customer_id=CURRENT_CUSTOMER_ID) # Force string ID
        return {"tool_name": tool_name, "result": result}
    
    if tool_name == "get_customer_products":
        result = get_customer_products(customer_id=CURRENT_CUSTOMER_ID, include_closed=tool_args.get("include_closed", False)) # Force string ID
        return {"tool_name": tool_name, "result": result}

    # 2. Tools that *do* require product_id
    if tool_name in ["get_account_balance", "get_recent_transactions", "get_transactions_by_date_range"]:
        
        # product_id_from_model is now a string (e.g., "2001" or "current account")
        product_id_from_model = tool_args.get("product_id")

        # Basic check to see if it is a valid ID-like format (4 digits)
        if product_id_from_model and product_id_from_model.isdigit() and len(product_id_from_model) == 4:
            correct_product_id_str = product_id_from_model # It is already a string!
            print(f"  -> [Brain] Model provided a valid product_id string: '{correct_product_id_str}'")
        else:
            # --- ID is missing or invalid (e.g., "current account"): Ask for clarification ---
            print(f"  -> [Brain] Model did not provide a valid product_id. Asking for clarification...")
            products = products_df[
                (products_df['customer_id'] == CURRENT_CUSTOMER_ID) & 
                (products_df['product_type'].isin(['Current Account', 'Savings Account']))
            ]
            if products.empty:
                return {"tool_name": tool_name, "result": json.dumps({"error": "You have no accounts that support transactions."})}
            
            product_names = [f"{p['product_name']} (product ID {p['product_id']})" for index, p in products.iterrows()]
            error_message = f"You have multiple accounts: {', '.join(product_names)}. Which account would you like information about? Please specify the product ID."
            
            return {"tool_name": tool_name, "result": json.dumps({"error": error_message})}

        # --- ID is present and is a string: Proceed ---
        if tool_name == "get_account_balance":
            result = get_account_balance(product_id=correct_product_id_str)
        elif tool_name == "get_recent_transactions":
            result = get_recent_transactions(product_id=correct_product_id_str, limit=tool_args.get("limit", 5))
        elif tool_name == "get_transactions_by_date_range":
            result = get_transactions_by_date_range(
                product_id=correct_product_id_str, 
                start_date=tool_args["start_date"], 
                end_date=tool_args["end_date"]
            )
        else:
             result = json.dumps({"error": "Internal error, tool not found after ID check."})
        
        return {"tool_name": tool_name, "result": result}
    
    # Unknown tool
    return None


def answer_my_question(full_chat_history: list) -> str:
    """(Brain-Orchestrator) This is the main brain. It receives the *entire* chat history and decides what to do."""
    
    api_history = []
    for msg in full_chat_history:
        api_history.append({"role": msg["role"], "parts": [msg["content"]]})

    print(f"\n--- [Brain] Processing... (Input: {len(api_history)} messages) ---")
    
    # [!!] The Super-Strict System Prompt
    system_prompt = f"""
    You are an ING bank assistant for Customer ID {CURRENT_CUSTOMER_ID}.
    
    IMPORTANT CONTEXT: The current year is 2025. Use 2025 for all dates (e.g., "October 1 to 5" means "October 1, 2025" to "October 5, 2025").
    
    YOUR TASK IS TO CHOOSE A PATH: "TOOL" OR "RAG".
    
    PATH 1: USE A TOOL (VERY STRICT)
    If the *last question* from the user is about THEIR personal banking, YOU MUST call one of these tools:
    - `get_customer_details` (for address, phone, etc.)
    - `get_customer_products` (for "my accounts", "my products")
    - `get_account_balance` (for "balance", "how much money")
    - `get_recent_transactions` (for "recent transactions", "what have I done")
    - `get_transactions_by_date_range` (for "spending between", "how much did I spend")
    
    PATH 2: USE RAG (Fallback)
    If, and *ONLY* if, the user's question is a GENERAL, NON-PERSONAL banking question (e.g., "How do I block a card?"), YOU MUST call the `answer_with_rag` function.
    
    RULE 3: NEVER MAKE UP ANSWERS (STRICT)
    NEVER generate an answer about the customer's personal banking yourself. (FORBIDDEN: "I don't have that information", "I can only tell you about...").
    
    Analyze the ENTIRE HISTORY to understand context (e.g., which product ID is meant).
    """
    
    api_history_with_prompt = [{"role": "user", "parts": [{"text": system_prompt}]}]
    for msg in full_chat_history:
        role = "user" if msg["role"] == "user" else "model"
        api_history_with_prompt.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Step 1: Let the model decide
    first_response = tool_llm.generate_content(api_history_with_prompt)

    try:
        function_calls = first_response.candidates[0].function_calls
    except (IndexError, AttributeError):
        function_calls = []

    # Decision A: Use Tools
    if function_calls:
        print(f"  -> [Brain] Decision: Use {len(function_calls)} tool(s). (PATH 1 CHOSEN)")
        tool_results_parts_list = []
        
        for tool_call in function_calls:
            tool_result = execute_tool_call(tool_call)
            if tool_result:
                tool_results_parts_list.append(
                    {
                        "function_response": {
                            "name": tool_call.name,
                            "response": {"content": tool_result["result"]}
                        }
                    }
                )

        if not tool_results_parts_list:
            return "Sorry, I could not execute the tool call correctly."

        # Step 3: Send tool results back
        print(f"  -> [Brain] Returning {len(tool_results_parts_list)} tool results...")
        model_response_part_dict = first_response.candidates[0].content.parts[0].to_dict()
        message_history_for_final_call = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            *api_history_with_prompt[1:], 
            {"role": "model", "parts": [model_response_part_dict]}, 
            {"role": "user", "parts": tool_results_parts_list} 
        ]
        final_response = tool_llm.generate_content(message_history_for_final_call)
        return final_response.text.strip()
    
    # Decision B: Use RAG
    else:
        print(f"  -> [Brain] Decision: No tool chosen. Passing to RAG. (PATH 2 CHOSEN)")
        last_user_query = full_chat_history[-1]["content"]
        return answer_with_rag(last_user_query)

# ===============================================
# --- 4. Web "Mouth" (TTS) ---
# ===============================================
@st.cache_data
def synthesize_speech(text_to_speak: str) -> bytes:
    print(f"[TTS]: Synthesizing speech for... '{text_to_speak[:20]}...'")
    client = texttospeech.TextToSpeechClient()
    synthesis_input = SynthesisInput(text=text_to_speak)
    voice = VoiceSelectionParams(
        language_code="nl-BE", 
        name="nl-BE-Wavenet-B"
    )
    audio_config = AudioConfig(audio_encoding=AudioEncoding.MP3)
    response = client.synthesize_speech(
        request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
    )
    return response.audio_content

# ===============================================
# --- 5. Web "Ears" (STT) ---
# ===============================================
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Convert stereo audio bytes to mono and send to Google STT."""
    print(f"[STT]: Received {len(audio_bytes)} bytes of audio...")
    try:
        stereo_sound = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        mono_sound = stereo_sound.set_channels(1)
        mono_bytes_io = io.BytesIO()
        mono_sound.export(mono_bytes_io, format="wav")
        mono_audio_bytes = mono_bytes_io.getvalue()
    except Exception as e:
        print(f"[STT] Audio conversion error: {e}")
        st.error(f"Audio processing failed: {e}")
        return ""

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=mono_audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="nl-BE"
    )
    try:
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            print("[STT]: No speech detected.")
            return ""
        transcript = response.results[0].alternatives[0].transcript
        print(f"[STT]: Transcription result: {transcript}")
        return transcript
    except Exception as e:
        print(f"[STT] Error: {e}")
        st.error(f"Speech recognition failed: {e}")
        return ""

# ===============================================
# --- 6. Streamlit Web Interface ---
# ===============================================

st.set_page_config(page_title="ING Voice Assistant", layout="centered")
st.title("ING Voice Assistant 🎙️")
st.markdown(f"*(Demo logged in as Customer ID: **{CURRENT_CUSTOMER_ID}**)*")

# --- [!!] Fix: Initialize all session state variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_data" not in st.session_state:
    st.session_state.last_audio_data = None  # <--- Critical state variable

# 1. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("audio"):
            # Check autoplay flag
            st.audio(message["audio"], format="audio/mp3", autoplay=message.get("autoplay", False))
            # [!!] Critical: Reset autoplay flag after playing to prevent replay on re-run
            if message.get("autoplay"):
                message["autoplay"] = False 

# 2. Render recording component
wav_audio_data = st_audiorec()

# 3. Core processing loop
# [!!] Critical fix: Check if this is *new* audio
if wav_audio_data is not None and wav_audio_data != st.session_state.last_audio_data:
    
    # a. Mark this audio as "processed" to prevent looping
    st.session_state.last_audio_data = wav_audio_data
    
    # b. STT (ears)
    user_transcript = transcribe_audio_bytes(wav_audio_data)

    if user_transcript:
        # c. Add user message to the UI
        st.session_state.messages.append({"role": "user", "content": user_transcript})
        
        # d. "Think" (Brain) + Loading indicator
        with st.spinner("One moment, I am looking it up..."):
            # [!!] Critical fix: Pass the *entire* chat history, not just the last message
            assistant_response_text = answer_my_question(st.session_state.messages)
            assistant_response_audio = synthesize_speech(assistant_response_text)
        
        # e. (For diagnostics) Save file
        try:
            with open("tts_output.mp3", "wb") as f:
                f.write(assistant_response_audio)
            print("[Diagnostics]: tts_output.mp3 file successfully saved!")
        except Exception as e:
            print(f"[Diagnostics]: Failed to save tts_output.mp3 file: {e}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": assistant_response_text, 
            "audio": assistant_response_audio,
            "autoplay": True
        })
        
        st.rerun() # Restart the script to display new messages
        
    else:
        st.warning("I couldn't understand you. Could you try again?")

elif not st.session_state.messages:
    st.info("Click the microphone to start recording.")
