#Edureko Agentic AI Module 5

A modular, intelligent competitive analysis system that uses Retrieval-Augmented Generation (RAG) with agentic reasoning capabilities to analyze competitor data and provide strategic insights.

## 🚀 Features

- **Agentic RAG with ReAct Framework**: Implements reasoning and action cycles for intelligent query processing
- **CSV Data Loading**: Loads competitor data from CSV files for real-world applicability
- **Intent Analysis**: Automatically detects query intent and adapts response strategy
- **Comparison Capabilities**: Performs detailed competitive comparisons
- **Query History**: Maintains searchable history of queries and responses
- **Modular Architecture**: Clean, maintainable code structure with separation of concerns
- **Interactive CLI**: User-friendly command-line interface with help system

## 📁 Project Structure

```
competitive_analysis_agent/
├── config.py              # Configuration and constants
├── data/
│   └── competitor_data.csv # CSV data file (auto-created)
├── data_processor.py      # Data loading and preprocessing
├── vector_store.py        # Vector indexing with Cohere embeddings
├── query_analyzer.py      # Query intent analysis and planning
├── react_agent.py         # ReAct framework implementation
├── history_manager.py     # Query history management
├── cli_interface.py       # Interactive command-line interface
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## 🛠️ Installation

1. **Create the project directory:**
```bash
mkdir competitive_analysis_agent
cd competitive_analysis_agent
```

2. **Save each module as separate Python files** with the names shown in the project structure

3. **Create the data directory:**
```bash
mkdir data
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your Cohere API key
```

6. **Run the application:**
```bash
python main.py
```

## 🔧 Configuration

### Environment Variables (.env file)
```
COHERE_API_KEY=your_cohere_api_key_here
```

### CSV Data Format
The system expects a CSV file with the following columns:
- `Competitor Name`: Name of the competitor company
- `Product Description`: Description of their products/services
- `Marketing Strategy`: Their marketing approach and tactics
- `Financial Summary`: Revenue, growth, and financial information

If no CSV file exists, the system will automatically create a sample file with demo data.

## 💡 Usage Examples

### Query Types Supported:
- **General Information**: "Tell me about TechCorp"
- **Comparisons**: "Compare TechCorp and InnovateLabs marketing strategies"
- **Financial Analysis**: "Which company has the highest revenue?"
- **Product Analysis**: "Analyze the AI capabilities of different companies"
- **Market Insights**: "What are the main trends in marketing strategies?"

### Available Commands:
- `help` - Show detailed help information
- `history` - View recent queries and responses
- `clear` - Clear query history
- `export` - Export history to a text file
- `exit` - Quit the application

## 🏗️ Architecture

### Modular Design
Each module has a specific responsibility:

- **config.py**: Centralized configuration management
- **data_processor.py**: CSV loading, validation, and preprocessing
- **vector_store.py**: Vector indexing and embedding management
- **query_analyzer.py**: Intent detection and query planning
- **react_agent.py**: Core reasoning and action implementation
- **history_manager.py**: Query history storage and retrieval
- **cli_interface.py**: User interaction and command processing
- **main.py**: Application coordination and error handling


## 🔍 Technical Details

### Models Used:
- **Embeddings**: Cohere's `embed-english-v3.0`
- **Generation**: Cohere's `command-r-plus`
- **Vector Store**: LlamaIndex VectorStoreIndex

### Key Features:
- Adaptive similarity search based on query complexity
- Reasoning process transparency and logging
- Automatic competitor name extraction and matching
- Intent-based response enhancement
- Error handling and graceful degradation

## 🚀 Extending the System

The modular architecture makes it easy to extend:

- **Add new data sources**: Modify `data_processor.py`
- **Implement different vector stores**: Extend `vector_store.py`
- **Add new query types**: Enhance `query_analyzer.py`
- **Create web interface**: Replace `cli_interface.py`
- **Add new reasoning strategies**: Extend `react_agent.py`

## 📊 Sample Data

The system includes sample data for 10 technology companies with information about:
- Cloud computing platforms
- AI and ML solutions
- Data processing tools
- Cybersecurity services
- Digital transformation consulting

## 🛡️ Error Handling

The system includes comprehensive error handling:
- API connection issues
- CSV file validation
- Missing dependencies detection
- Graceful degradation on errors
- Detailed logging for debugging

## 📝 Getting Your Cohere API Key

1. Visit [Cohere's website](https://cohere.ai/)
2. Sign up for an account or log in
3. Navigate to the API keys section in your dashboard
4. Generate a new API key
5. Add it to your `.env` file

## 🔐 Security Best Practices

- Never commit your `.env` file to version control
- Keep your API keys secure and rotate them regularly
- Use environment variables for all sensitive configuration
- Validate all user inputs before processing
