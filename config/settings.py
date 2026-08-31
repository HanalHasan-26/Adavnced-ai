from pathlib import Path


# =========================================================
# APPLICATION PATHS
# =========================================================

# Root directory of the project.
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory for persistent application data.
DATA_DIR = BASE_DIR / "data"

# Directory for application logs.
LOGS_DIR = BASE_DIR / "logs"


# =========================================================
# APPLICATION INFORMATION
# =========================================================

APP_NAME = "Advanced AI"

APP_VERSION = "0.1.0"


# =========================================================
# LOCAL LLM
# =========================================================

# Ollama model used by the assistant.
MODEL_NAME = "qwen3:1.7b"

# Ollama server address.
OLLAMA_BASE_URL = "http://localhost:11434"

# Maximum time to wait for an Ollama request.
OLLAMA_TIMEOUT = 120.0


# =========================================================
# KNOWLEDGE SYSTEM
# =========================================================

# SQLite database containing the knowledge base.
KNOWLEDGE_DATABASE_PATH = (
    DATA_DIR / "knowledge.db"
)

# Default number of knowledge results retrieved.
DEFAULT_RETRIEVAL_LIMIT = 5


# =========================================================
# CONVERSATION MEMORY
# =========================================================

# SQLite database containing conversation memory.
MEMORY_DATABASE_PATH = (
    DATA_DIR / "memory.db"
)

# Number of previous conversation entries used
# when building context.
DEFAULT_MEMORY_LIMIT = 5


# =========================================================
# WEB RESEARCH
# =========================================================

# Default number of web search results.
DEFAULT_WEB_SEARCH_LIMIT = 5


# =========================================================
# LOGGING
# =========================================================

# Default application log level.
LOG_LEVEL = "INFO"