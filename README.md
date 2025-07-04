# LLM Factory: Unified LLM & Document Parsing API

LLM Factory is a modular Python framework for interacting with Large Language Models (LLMs) from multiple providers (Azure OpenAI, OpenAI, Groq, etc.) and for parsing documents (PDF, DOCX, PPTX) using a unified interface.

---

## 📦 Installation

```bash
pip install -e git+https://github.com/shryesth/llm_factory.git#egg=llm_factory
```

Or, for a specific version if published:

```bash
pip install llm_factory
```

---

## 🔑 Environment Variables

- Place your `.env` file in the root of your project (the project that *uses* `llm_factory`).
- The package will automatically attempt to load environment variables from the current working directory's `.env` file upon import.
- See `.env.example` in the `llm_factory` project root for all configuration options.

---

## 🛠️ Exposed API

You can import the following directly from `llm_factory`:

```python
from llm_factory import (
    # Core
    run_pipeline,           # Main function to run LLM prompt pipelines
    LLMClientFactory,       # Factory for creating LLM clients
    PromptProcessorFactory, # Factory for creating prompt processors
    ParserFactory,          # Factory for document parsers (PDF, DOCX, PPTX, etc.)
    ENV_VARS,               # Dictionary of loaded environment variables
    get_client_config,      # Function to get specific client config

    # Base Classes (for type hinting and extension)
    LLMClient,              # Abstract base class for LLM clients (from core)
    BaseLLMClient,          # Base class for specific LLM client implementations
    PromptProcessor,        # Abstract base class for prompt processors (from core)
    BasePromptProcessor,    # Base class for specific prompt processor implementations
    BaseParser,             # Base class for specific document parser implementations

    # Concrete LLM Clients
    AzureLLMClient,
    GroqLLMClient,
    LMStudioClient,
    OllamaLLMClient,
    OpenAILLMClient,

    # Concrete Parsers
    PDFParser,
    DocxParser,
    PPTProcessor,           # Parser for PPTX files

    # Concrete Prompt Processors
    StandardPromptProcessor,
    ChainOfThoughtProcessor
)
```

---

## 🆕 Enhanced Configuration Features

### Model Parameters and History Controls

LLM Factory now supports advanced configuration options for both CoT (Chain of Thought) and Standard pipelines:

### Fine-Tuning Enhancement

**NEW**: Automatic prompt fine-tuning using AI-powered prompt enhancement. When `fine_tune_prompt` is included in your pipeline configuration, each step's prompt will be automatically enhanced before execution using the specified guidelines.

#### How Fine-Tuning Works
- **Automatic Enhancement**: Each step's prompt is analyzed and improved using AI
- **Context-Aware**: Enhancement considers previous steps and pipeline context
- **Caching**: Enhanced prompts are cached to avoid redundant processing
- **Dry-Run Support**: Fine-tuning respects dry-run mode and won't make API calls during testing
- **Client Type Respect**: Uses your configured default client type (e.g., lmstudio, azure, openai)

#### Top-Level Configuration
- **`model`**: Specify which model to use (overrides client default)
- **`temperature`**: Control randomness in responses
- **`max_tokens`**: Set maximum response length
- **`dry_run`**: Enable simulation mode (logs payloads without API calls)
- **`fine_tune_prompt`**: Enable automatic prompt enhancement with custom guidelines

#### Step-Level Overrides
- Any step can override top-level settings
- **`exclude_from_chat_history`**: Prevent step from being added to conversation history
- **`exclude_context`**: Exclude context data from specific steps

#### History Management
- Fine-grained control over what gets stored in conversation history
- Useful for large contexts or sensitive information
- Maintains conversation flow while optimizing token usage

### Example Enhanced Configuration

```python
# Enhanced CoT Pipeline with all new features including fine-tuning
cot_config = {
    "name": "problem_solver",
    "model": "gpt-4",           # Top-level model setting
    "temperature": 0.7,         # Top-level temperature
    "max_tokens": 800,          # Top-level token limit
    "dry_run": False,           # Set to True for simulation
    "context_data": "Very long context like a book",
    # NEW: Fine-tuning enhancement
    "fine_tune_prompt": """
    Focus on creating detailed, analytical responses that break down complex problems
    into manageable steps. Use clear reasoning and provide concrete examples.
    Emphasize practical solutions over theoretical discussions.
    """,
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": "Summarize the problem from this context.",
            "output_key": "problem_summary",
            "exclude_context": True  # Don't include context in this step
        },
        {
            "type": "initialPrompt",
            "name": "generate_hypothesis",
            "prompt": "Based on the summary, what hypotheses can you derive?",
            "input_key": "problem_summary",
            "output_key": "hypotheses",
            "temperature": 0.3,      # Override temperature for this step
            "exclude_from_chat_history": True  # Don't store in history
        },
        {
            "type": "finalAnswer",
            "name": "conclusion",
            "prompt": "Give your final answer.",
            "input_key": "hypotheses",
            "output_key": "conclusion"
            # Uses top-level settings
        }
    ]
}

# Enhanced Standard Processor Configuration
standard_config = {
    "name": "data_extraction",
    "model": "claude-3-sonnet",  # Use different model
    "temperature": 0.1,          # Low temperature for precision
    "dry_run": True,             # Test mode
    "prompt": "Extract key information from the document.",
    "context_data": "Document content here...",
    "exclude_context": False,    # Include context
    "schema": {                   # JSON schema for structured output
        "type": "object",
        "properties": {
            "key_points": {"type": "array", "items": {"type": "string"}}
        }
    }
}
```

### Dry-Run Mode

Enable dry-run mode to test your pipeline configurations without making actual API calls:

```python
from llm_factory import run_pipeline

# Test your configuration
result = run_pipeline(
    prompt_config={
        "name": "test_pipeline",
        "dry_run": True,  # This will log all payloads without API calls
        "model": "gpt-4",
        "temperature": 0.5,
        "steps": [...]
    },
    client_type="openai",
    pipeline_type="multi_step"
)
```

#### Dry-Run Features

**Console Logging**: Dry-run mode will log to console:
- Model being used
- Temperature and token settings
- Complete message payloads
- Schema information
- History control settings
- Fine-tuning status and cache information

**File Logging**: All dry-run requests are automatically saved to timestamped JSON files:
- **Folder**: `llm_factory_dry_run_responses/` (created automatically)
- **Individual Requests**: `YYYYMMDD_HHMMSS_###_pipeline_step_type.json`
- **Pipeline Summaries**: `YYYYMMDD_HHMMSS_SUMMARY_pipeline.json`
- **Complete Results**: `YYYYMMDD_HHMMSS_RESULT_pipeline.json` *(NEW)*
- **Fine-Tuning Details**: `YYYYMMDD_HHMMSS_FINETUNE_pipeline_step.json` *(NEW)*

**Example Files Created**:
```
llm_factory_dry_run_responses/
├── 20250704_143022_001_problem_solver_understand_problem_initialPrompt.json
├── 20250704_143022_002_problem_solver_generate_hypothesis_newProblem.json
├── 20250704_143022_003_problem_solver_conclusion_finalAnswer.json
├── 20250704_143022_SUMMARY_problem_solver.json
├── 20250704_143022_RESULT_problem_solver.json  ← Complete pipeline result
├── 20250704_143022_FINETUNE_problem_solver_understand_problem.json  ← Fine-tuning details
├── 20250704_143022_FINETUNE_problem_solver_generate_hypothesis.json
└── 20250704_143022_FINETUNE_problem_solver_conclusion.json
```

**Individual Request File Structure**:
```json
{
  "session_info": {
    "session_id": "20250704_143022",
    "request_number": 1,
    "timestamp": "2025-07-04T14:30:22.123456"
  },
  "pipeline_info": {
    "pipeline_name": "problem_solver",
    "step_name": "understand_problem",
    "step_type": "initialPrompt"
  },
  "request_data": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 800,
    "messages": [...],
    "response_format": {...}
  },
  "metadata": {
    "exclude_from_history": false,
    "exclude_context": true,
    "step_index": 0,
    "fine_tuned": false
  }
}
```

**Complete Result File Structure** *(NEW)*:
```json
{
  "session_info": {
    "session_id": "20250704_143022",
    "timestamp": "2025-07-04T14:30:22.123456",
    "log_type": "complete_pipeline_result"
  },
  "pipeline_info": {
    "pipeline_name": "problem_solver",
    "total_requests_logged": 3
  },
  "pipeline_config": {
    "name": "problem_solver",
    "model": "gpt-4",
    "temperature": 0.7,
    "dry_run": true,
    "fine_tune_prompt": "...",
    "steps": [...]
  },
  "complete_result": {
    "problem_solver": {
      "success": true,
      "steps": {...},
      "conversation_history": "...",
      "json_result": {...},
      "tonality_result": {...},
      "final_output": {...},
      "history_messages": [...],
      "pipeline_config": {...}
    }
  }
}
```

**Fine-Tuning Process File Structure** *(NEW)*:
```json
{
  "session_info": {
    "session_id": "20250704_143022",
    "timestamp": "2025-07-04T14:30:22.123456",
    "log_type": "fine_tuning_process"
  },
  "pipeline_info": {
    "pipeline_name": "problem_solver",
    "step_name": "understand_problem"
  },
  "fine_tuning_process": {
    "original_prompt": "Analyze this math problem step by step...",
    "fine_tuning_guidelines": "Focus on creating detailed, analytical responses that break down complex problems into manageable steps...",
    "enhanced_prompt": "You are an expert mathematical analyst tasked with solving complex problems through systematic breakdown...",
    "prompt_length_comparison": {
      "original_length": 156,
      "enhanced_length": 847,
      "length_increase": 691
    }
  },
  "enhancement_metadata": {
    "step_index": 0,
    "cache_hit": false,
    "enhancement_type": "general",
    "context_provided": true
  }
}
```

---

## 📖 Usage Examples

(Refer to scripts in the `examples/` directory for more detailed usage.)

### 1. Running a Prompt Pipeline

```python
from llm_factory import run_pipeline

# Ensure your .env file is in the current working directory or environment variables are set
result = run_pipeline(
    prompt_config={
        "name": "simple_prompt",
        "prompt": "What is the capital of France?"
    },
    client_type="openai",      # or "azure", "groq", "lmstudio", "ollama"
    pipeline_type="standard",  # or "cot" (chain-of-thought) / "multi_step"
    temperature=0.2,
    max_tokens=1000
)
print(result)
# Expected: Paris (or similar)
```

### 2. Using a Specific LLM Client (e.g., OpenAI)

```python
from llm_factory import OpenAILLMClient, LLMClientFactory

# Option 1: Direct instantiation
# Ensure OPENAI_API_KEY is set in your environment or .env file
try:
    client = OpenAILLMClient()
    response = client.generate_completion(
        messages=[{"role": "user", "content": "Tell me a joke."}]
    )
    print(f"Direct Client Response: {response}")
except RuntimeError as e:
    print(f"Error instantiating client: {e}")


# Option 2: Using the factory
# Ensure relevant environment variables for 'openai' are set
try:
    factory_client = LLMClientFactory.create_client(client_type="openai")
    response_factory = factory_client.generate_completion(
        messages=[{"role": "user", "content": "What is 1 + 1?"}]
    )
    print(f"Factory Client Response: {response_factory}")
except ValueError as e:
    print(f"Error creating client via factory: {e}")
except RuntimeError as e:
    print(f"Error during client operation: {e}")
```

### 3. Using the Document Parser Factory (e.g., PDF)

```python
from llm_factory import ParserFactory, PDFParser
import os # Added for os.path.exists

# Example with PDFParser directly
# pdf_parser = PDFParser(file_path_or_url="/path/to/your/document.pdf")
# content = pdf_parser.parse()
# print(f"PDF Content (Direct): {content[:500]}...") # Print first 500 chars

# Example with ParserFactory
try:
    # Replace with an actual path to a PDF or DOCX file for testing
    # Ensure the file exists at the specified path
    # For PDF: requires 'pip install pymupdf'
    # For DOCX: requires 'pip install python-docx'
    # For PPTX: requires 'pip install python-pptx'
    
    # file_to_parse = "/path/to/your/file.pdf" 
    # file_to_parse = "/path/to/your/file.docx"
    file_to_parse = "example.pdf" # Create a dummy example.pdf or use a real one
    
    # Create a dummy PDF for the example to run without external files
    try:
        from llm_factory.parsers.pdf.parser import PDFParser # To check if available
        # This is a simplified way to create a dummy PDF; real PDFs are more complex.
        # For a real test, use an actual PDF file.
        # If PyMuPDF is not installed, this part will be skipped.
        import fitz # PyMuPDF
        doc = fitz.open() # New empty PDF
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test PDF document for llm_factory.")
        doc.save(file_to_parse)
        doc.close()
        print(f"Created dummy '{file_to_parse}' for example.")
    except ImportError:
        print(f"PyMuPDF (fitz) not installed, skipping dummy PDF creation. Parser example might fail if '{file_to_parse}' does not exist.")
    except Exception as e:
        print(f"Could not create dummy PDF: {e}")


    if os.path.exists(file_to_parse):
        parser = ParserFactory.create_parser(file_path_or_url=file_to_parse)
        if parser:
            parsed_content = parser.parse() 
            print(f"Parsed Content (Factory for '{file_to_parse}'): {parsed_content[:500]}...") # Print first 500 chars
        else:
            print(f"Could not create a parser for '{file_to_parse}'. Check file type and dependencies.")
    else:
        print(f"File '{file_to_parse}' not found. Skipping parser example.")

except ImportError as e:
    print(f"A required parsing library is not installed: {e}. Skipping parser example.")
except Exception as e:
    print(f"An error occurred in the parser example: {e}")

```

### 4. Using a Specific Prompt Processor (e.g., StandardProcessor)

```python
from llm_factory import StandardPromptProcessor, OpenAILLMClient, LLMClientFactory

# Ensure your .env file is in the current working directory or environment variables are set
try:
    # Get a client (e.g., OpenAI)
    # client = OpenAILLMClient() 
    # Or use the factory:
    client = LLMClientFactory.create_client(client_type="openai")

    # Initialize the processor
    processor = StandardPromptProcessor()

    # Define a prompt configuration
    prompt_config = {
        "name": "greeting_prompt",
        "prompt": "Generate a friendly greeting for a new user named Alex.",
        # "schema": {"type": "object", "properties": {"greeting": {"type": "string"}}} # Optional schema
    }

    # Process the prompt
    result = processor.process(
        client=client,
        prompt_config=prompt_config,
        temperature=0.7,
        max_tokens=50
    )
    print(f"Standard Processor Result: {result}")
    # Expected: {'name': 'greeting_prompt_processed', 'result': 'Hello Alex, welcome!'} (or similar)

except Exception as e:
    print(f"Error in Standard Processor example: {e}")
```

### 5. Accessing Environment Variables

```python
from llm_factory import ENV_VARS, get_client_config

# Access all loaded environment variables (filtered by llm_factory's interests)
# print(f"Loaded ENV_VARS: {ENV_VARS}")

# Get specific configuration for a client
openai_config = get_client_config("openai")
if openai_config:
    print(f"OpenAI API Key (from get_client_config): {openai_config.get('api_key')}")
else:
    print("OpenAI configuration not found.")

azure_config = get_client_config("azure")
if azure_config:
    print(f"Azure Endpoint (from get_client_config): {azure_config.get('azure_endpoint')}")
else:
    print("Azure configuration not found.")
```

---

## 🔧 Extending the Factory

### Adding New LLM Providers

1. Create a new client class in `llm_factory/clients/` inheriting from `BaseLLMClient`.

    ```python
    # llm_factory/clients/new_provider_client.py
    from .base_client import BaseLLMClient

    class NewProviderClient(BaseLLMClient):
        # Implement abstract methods
        def generate_completion(self, messages, **kwargs):
            pass
    ```

2. Update `llm_factory/core.py` in `LLMClientFactory.create_client` to include the new client.

    ```python
    # llm_factory/core.py (inside LLMClientFactory)
    elif client_type.lower() == "new_provider":
        from .clients.new_provider_client import NewProviderClient
        return NewProviderClient(**kwargs)
    ```

3. Add necessary environment variables to your `.env` file and document them in `.env.example`.

### Adding New Document Parsers

1. Create a new parser class in `llm_factory/parsers/your_format/` inheriting from `BaseParser`.

    ```python
    # llm_factory/parsers/your_format/parser.py
    from ..base_parser import BaseParser

    class YourFormatParser(BaseParser):
        def parse(self, **kwargs) -> str:
            pass
    ```

2. Update `llm_factory/parsers/factory.py` in `ParserFactory.create_parser` to include the new parser.

    ```python
    # llm_factory/parsers/factory.py (inside ParserFactory)
    elif ext == ".your_ext":
        from .your_format.parser import YourFormatParser
        return YourFormatParser(file_path_or_url, **kwargs)
    ```

---

## 📁 Current Project Structure

The `llm_factory` project and package are organized as follows:

```
llm_factory/                     # Root of the llm_factory project
├── llm_factory/                 # Main package source code
│   ├── __init__.py              # Initializes the package, exposes API
│   ├── core.py                  # Core logic: run_pipeline, LLMClientFactory, PromptProcessorFactory
│   ├── env_loader.py            # Loads and manages environment variables
│   ├── clients/                 # LLM client implementations
│   │   ├── __init__.py
│   │   ├── base_client.py       # Abstract base class for LLM clients
│   │   ├── azure_client.py
│   │   ├── groq_client.py
│   │   ├── lmstudio_client.py
│   │   ├── ollama_client.py
│   │   └── openai_client.py
│   ├── parsers/                 # Document parsing implementations
│   │   ├── __init__.py
│   │   ├── base_parser.py       # Abstract base class for parsers
│   │   ├── factory.py           # ParserFactory for creating parser instances
│   │   ├── docx/
│   │   │   ├── __init__.py
│   │   │   └── parser.py        # DOCX parser
│   │   ├── pdf/
│   │   │   ├── __init__.py
│   │   │   ├── chunking.py      # PDF chunking utilities
│   │   │   └── parser.py        # PDF parser
│   │   └── pptx/
│   │       ├── __init__.py
│   │       └── parser.py        # PPTX parser
│   ├── processors/              # Prompt processing strategies
│   │   ├── __init__.py
│   │   ├── base_processor.py    # Abstract base class for prompt processors
│   │   ├── cot_processor.py     # Chain of Thought processor
│   │   └── standard_processor.py # Standard prompt processor
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── image_utils.py       # Image processing utilities
├── examples/                    # Usage examples and test scripts
│   ├── __init__.py
│   ├── actual_structures.py
│   ├── example_parser_vision.py
│   ├── example_parser.py
│   ├── lmstudio_vision_example.py
│   ├── main.py                  # Main example script
│   ├── testing_llm_with_actual_documente.py
│   └── testing_structured_output_llm.py
├── .env.example                 # Example environment variable file
├── MANIFEST.in                  # Specifies files to include in source distributions
├── pyproject.toml               # Build system configuration
├── README.md                    # This file!
├── requirements.txt             # Project dependencies (for development)
└── setup.py                     # Script for packaging and distribution
```

---

## ❓ Troubleshooting

### Common Issues

1. **`ImportError: No module named 'dotenv'`**
    - Solution: Ensure `python-dotenv` is installed (`pip install python-dotenv`). It's listed in `setup.py`, so `pip install -e .` should handle it.
2. **`KeyError: 'azure'` (or other client type)**
    - Solution: Check your `.env` file to ensure all required environment variables for the selected client type are correctly set and loaded.
3. **`Failed to generate completion for 'X' after Y attempts.`**
    - Solution: This could be due to API errors, network issues, or misconfiguration. Check API keys, endpoints, model names, and rate limits for your LLM provider.
4. **`No schema matches the provided JSON`**
    - Solution: If using schema validation, ensure your JSON schema definition matches the expected output structure from the LLM.

### Enhanced Features Troubleshooting

5. **Model switching not working**
    - Solution: Ensure your client supports the specified model. Some models may not be available on all providers.
6. **Dry-run mode not showing expected output**
    - Solution: Check that `dry_run: true` is set in your configuration. Dry-run mode will return mock responses instead of actual API calls.
7. **History controls not working as expected**
    - Solution: Verify that `exclude_from_chat_history` and `exclude_context` are properly set at the step level. These only affect individual steps, not the entire pipeline.
8. **Step-level overrides not applying**
    - Solution: Ensure step-level parameters (`model`, `temperature`, `max_tokens`) are placed at the same level as other step properties like `prompt` and `type`.
9. **Dry-run files not being created**
    - Solution: Check that `dry_run: true` is set in your configuration. Ensure you have write permissions in the current directory. Files are saved to `llm_factory_dry_run_responses/` folder.
10. **Too many dry-run files accumulating**
    - Solution: The system creates a new session ID for each run. You can safely delete old files from `llm_factory_dry_run_responses/` folder. Each session is uniquely timestamped.
11. **Fine-tuning using wrong client type**
    - Solution: Ensure your `DEFAULT_CLIENT_TYPE` is set correctly in your `.env` file. The fine-tuning system now respects your default client configuration.
12. **Fine-tuning making API calls during dry-run**
    - Solution: This has been fixed. Fine-tuning now respects dry-run mode and will simulate enhancements without making actual API calls.

### Debug Mode

Enable more detailed logging by setting the `LOG_LEVEL` environment variable (e.g., in your `.env` file):

```env
LOG_LEVEL="DEBUG"
```

For enhanced debugging with the new features, enable dry-run mode in your pipeline configuration:

```python
# Enable dry-run for detailed payload inspection
pipeline_config = {
    "dry_run": True,  # This will log all request details
    # ... other configuration
}
```

**Analyzing Dry-Run Files**:
- Use any JSON viewer to inspect the generated files
- Each file contains the complete request that would be sent to the API
- Summary files provide overview of entire pipeline execution
- Files are organized by timestamp for easy chronological analysis

---

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-amazing-feature`).
3. Make your changes.
4. Add tests for your changes if applicable.
5. Ensure your code follows the project's coding style and includes appropriate documentation.
6. Commit your changes (`git commit -m 'Add some amazing feature'`).
7. Push to the branch (`git push origin feature/your-amazing-feature`).
8. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---
