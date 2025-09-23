"""
Configuration and constants for the Competitive Analysis Agent
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "embed-english-v3.0"
LLM_MODEL = "command-r-plus-08-2024"

# Application Configuration
MAX_HISTORY_SIZE = 5
SIMILARITY_TOP_K = 5
CSV_FILE_PATH = "data/competitor_data.csv"

# Cohere Optimization Settings for command-r-plus-08-2024
COHERE_TEMPERATURE = 0.1  # Low temperature for focused, factual responses
COHERE_MAX_TOKENS = 512   # Concise responses - optimal for competitive analysis
COHERE_P = 0.75          # Nucleus sampling for quality

# Embedding Settings for embed-english-v3.0
EMBEDDING_DIMENSIONS = 1024  # Native dimension for embed-english-v3.0
EMBEDDING_TRUNCATE = "END"   # Truncate from end if text too long

# Prompt Templates for command-r-plus-08-2024
SYSTEM_PROMPT = """You are a competitive analysis expert. Provide clear, factual responses about competitors based on the provided data. Be direct and concise."""

QUERY_PROMPT_TEMPLATE = """Based on the competitor data provided, answer this query directly and concisely:

Query: {query_str}

Context: {context_str}

Instructions:
- Be factual and specific
- Use data from the context
- Keep responses focused
- Avoid unnecessary elaboration

Answer:"""

# Validation
if not COHERE_API_KEY:
    raise ValueError("Please set your COHERE_API_KEY in the .env file")