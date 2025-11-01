## ING Voice Assistant - Technical Summary

### Problem Statement

The objective was to develop a proof-of-concept voice-enabled assistant for ING Belgium customers. The assistant needed to:

  * Understand and respond to customer queries in **Dutch (nl-BE)**.
  * Leverage both **general banking information** (from website content chunks) and **specific customer data** (from synthetic CSV files).
  * Guide users and provide information on products, transactions, and balances using natural language.
  * Distinguish between requests requiring general knowledge and those needing access to personal data.

### Solution Architecture

The system follows a voice-in, voice-out pipeline orchestrated by a central "Brain" component that decides whether to retrieve general information (RAG) or execute specific data lookups (Tools/Function Calling).

```mermaid
graph TD
    A[User Voice] --> B(Cloud STT API);
    B -- Transcribed Text --> C{Brain (LLM + Logic)\nDecide: RAG or Tool?};

    C -- General Query? --> D[RAG Flow];
    D -- Query Vector --> E(Faiss Index\nChunks Embeddings);
    E -- Relevant Chunks --> F[RAG Prompt Builder];
    F -- Contextual Prompt --> G{LLM (Gemini)\nGenerate from context};

    C -- Personal Query? --> H[Tool Flow];
    H -- Tool Request --> I{LLM (Gemini + Tools)\nSelect & Format Tool Call};
    I -- Formatted Call --> J[execute_tool_call (Python)];
    J -- Lookup --> K(Pandas DataFrames\nLoaded from CSVs);
    K -- Results --> J;
    J -- Tool Result --> L{LLM (Gemini + Tools)\nGenerate from tool result};

    G -- Text Response --> M(Cloud TTS API);
    L -- Text Response --> M;
    M -- Synthesized Speech --> N[Assistant Voice];
```

**Key Components:**

  * **Frontend:** Streamlit web application (`streamlit_app.py`) with `st_audiorec` for browser-based audio input.
  * **STT/TTS:** Google Cloud Speech-to-Text and Text-to-Speech APIs (`google-cloud-speech`, `google-cloud-texttospeech`).
  * **Core Logic ("Brain"):** Google Gemini model (`gemini-2.5-flash` via `google-cloud-aiplatform`) leveraging Function Calling capabilities, guided by a sophisticated System Prompt and limited chat history.
  * **RAG Knowledge Base:** Faiss (`faiss-cpu`) vector index built from website content chunks embedded using `gemini-embedding-001`.
  * **Customer Data:** Pandas DataFrames loaded from CSV files (`customers.csv`, `products.csv`, `products_closed.csv`, `transactions.csv`), queried by Python functions.
  * **Audio Processing:** Pydub library for stereo-to-mono conversion before STT.

-----

### AI Implementation Details

  * **Speech Handling:**
      * STT configured for `nl-BE`, `48000Hz` sample rate, and `LINEAR16` encoding after Python-based stereo-to-mono conversion.
      * TTS uses `nl-BE` language and a WaveNet voice (`nl-BE-Wavenet-B`) for natural output.
  * **Core Logic (Orchestration):**
      * The `answer_my_question` function acts as the central orchestrator.
      * It sends the **limited recent chat history** (last \~4 turns) along with a **"Super-Strikte" System Prompt** (including Few-Shot examples) to the Gemini model configured with tools (`tool_llm`).
      * This prompt **forces** the model to choose between **PAD 1 (TOOL)** for personal queries or **PAD 2 (RAG)** for general queries, explicitly forbidding self-generation for personal data. It also sets the context year to **2025** to match transaction data.
  * **RAG Implementation:**
      * Uses `gemini-embedding-001` to create embeddings for website chunks (offline via `hackathon-ing-app.py`) and user queries (runtime).
      * Faiss `IndexFlatL2` stores chunk embeddings locally (`ing_chunks_nl.index`).
      * `answer_with_rag` function retrieves the top 3 relevant chunks based on L2 distance, constructs a prompt containing these chunks as context, and calls the base Gemini model (`llm`) for generation.
  * **Tool (Function Calling) Implementation:**
      * Five Python functions (`get_customer_details`, `get_customer_products`, `get_account_balance`, `get_recent_transactions`, `get_transactions_by_date_range`) are defined.
      * Corresponding `FunctionDeclaration` objects describe these functions (parameters, descriptions) to the `tool_llm`. **Crucially, all ID parameters (`customer_id`, `product_id`) are declared as `type: "string"`**.
      * CSV data is loaded into Pandas DataFrames with **`dtype=str` enforced for all ID columns** during loading (`load_models_and_data`).
      * The `execute_tool_call` function handles incoming tool calls from the LLM:
          * It **enforces** the use of `CURRENT_CUSTOMER_ID` (as a string).
          * It performs lookups on the DataFrames using **string comparisons** for IDs.
          * It handles cases where `product_id` is missing by querying available accounts and returning an error message asking for clarification.
          * It calculates balances (`get_account_balance`) dynamically by summing transactions.

-----

### Known Limitations

  * **LLM Reliability:** Despite strict prompting and context limiting, the LLM might still occasionally fail to follow instructions (e.g., choose RAG incorrectly). 