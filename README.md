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

### Debug Mode

Enable more detailed logging by setting the `LOG_LEVEL` environment variable (e.g., in your `.env` file):

```env
LOG_LEVEL="DEBUG"
```

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
