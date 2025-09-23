# Competitive Analysis Agent - Streamlit UI

## Running the Application

### Option 1: Streamlit Web Interface (Recommended)
```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Option 2: Command Line Interface with Query History
```bash
python main.py
```

#### New CLI Commands:
- `history` - View the last 5 queries and responses  
- `history 10` - View the last 10 queries (or any number 1-50)
- `clear` - Clear query history
- `export` - Export history to a timestamped file
- `stats` - Show query statistics (avg processing time, etc.)
- `status` - Show agent status and configuration
- `help` - Display help information
- `exit` - Quit the application

## Features

### Streamlit UI Features:
- **Interactive Web Interface**: Clean, modern UI for better user experience
- **Real-time Query Processing**: Submit queries and see results instantly
- **Advanced Query History**: View up to 50 recent queries with timestamps and processing times
- **Interactive History Controls**: Configurable number of entries to display
- **Export Functionality**: Download query history as text files
- **Query Statistics**: View processing time analytics and query counts
- **Responsive Design**: Works on desktop and mobile devices
- **Visual Feedback**: Loading spinners, success messages, and error handling
- **Enhanced Sidebar**: Agent status, history management, and statistics

### Core Functionality:
- **Agentic RAG**: ReAct framework for intelligent query processing
- **Competitive Analysis**: Analyze competitor data from CSV files
- **Vector Search**: Cohere embeddings for semantic search
- **Query Intent Analysis**: Understand and classify user queries
- **Multi-step Reasoning**: Break down complex queries into sub-goals
- **Persistent Query History**: Automatic storage of all queries and responses
- **Performance Tracking**: Monitor query processing times and statistics
- **Comprehensive Error Handling**: Robust error management with retry logic

## Usage Examples

Try these sample queries in the Streamlit interface:
- "Compare TechCorp and InnovateLabs marketing strategies"
- "What are the financial strengths of CloudFirst?"
- "Analyze the product offerings of AI companies"
- "Which company has the highest growth rate?"

## Dependencies

All required packages are listed in `requirements.txt` and include:
- streamlit (for web UI)
- cohere (for embeddings and LLM)
- llama-index (for RAG framework)
- pandas (for data processing)
- python-dotenv (for environment variables)