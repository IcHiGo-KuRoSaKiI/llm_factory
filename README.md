# LLM Factory: Unified LLM & Document Parsing API

LLM Factory is a modular Python framework for interacting with Large Language Models (LLMs) from multiple providers (Azure OpenAI, OpenAI, Groq, etc.) and for parsing documents (PDF, DOCX, PPTX) using a unified interface.

---

## Installation

```bash
pip install -e .
```

---

## Environment Variables

- Place your `.env` file in the root of your project (the project that uses `llm_factory`.
- The package will automatically load environment variables from the current working directory's `.env` file.
- See `.env.example` for configuration options.

---

## Exposed API

You can import the following directly from `llm_factory`:

```python
from llm_factory import (
    run_pipeline,           # Main function to run LLM prompt pipelines
    LLMClientFactory,       # Factory for creating LLM clients
    LLMClient,              # Abstract base class for LLM clients
    PromptProcessor,        # Abstract base class for prompt processors
    PromptProcessorFactory, # Factory for creating prompt processors
    PDFParser,              # PDF document parser
    ParserFactory           # Factory for document parsers (PDF, DOCX, PPTX, etc.)
)
```

---

## Usage Examples

### 1. Running a Prompt Pipeline

```python
from llm_factory import run_pipeline

result = run_pipeline(
    prompt_config={
        "name": "simple_prompt",
        "prompt": "What is the capital of France?"
    },
    client_type="openai",      # or "azure", "groq", etc.
    pipeline_type="standard",  # or "cot" for chain-of-thought
    temperature=0.2,
    max_tokens=1000
)
print(result)
```

### 2. Using the PDF Parser

```python
from llm_factory import PDFParser

# You need to provide an openai_helper (see your project for implementation)
pdf_parser = PDFParser(openai_helper)
parsed_content = pdf_parser.parse("/path/to/file.pdf")
print(parsed_content)
```

### 3. Using the Document Parser Factory

```python
from llm_factory import ParserFactory

parser = ParserFactory.create_parser("/path/to/file.pdf", openai_helper)
parsed_content = parser.parse("/path/to/file.pdf")
```

---

## Extending the Factory

- To add a new LLM provider, implement a new client in `clients/` and register it in `LLMClientFactory`.
- To add a new document parser, implement it in `parsers/` and register it in `ParserFactory`.

---

## License

MIT License

---

# LLM Factory: A Flexible Framework for LLM Interaction

## 🌟 Overview

LLM Factory is a modular, extensible framework for interacting with Large Language Models through various providers (Azure OpenAI, OpenAI, Groq, etc.) using a unified interface. It simplifies the process of working with LLMs by abstracting away provider-specific implementations, allowing you to focus on your prompts and applications.

### Key Features

- **Multiple LLM Provider Support**: Use Azure OpenAI, OpenAI, Groq, or add your own providers
- **Environment-based Configuration**: Easily switch between providers using environment variables
- **Two Processing Paradigms**:
  - **Standard Extraction**: Simple prompt-response workflow with optional JSON schema enforcement and tonality matching
  - **Chain of Thought (CoT)**: Multi-step prompting pipelines for complex reasoning tasks
- **Factory Pattern Implementation**: Cleanly separated interfaces for easy extension and testing
- **Conversation History Management**: Maintain context across multiple interactions

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [Extending the Factory](#-extending-the-factory)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- An API key for at least one LLM provider (Azure OpenAI, OpenAI, or Groq)

### Setup

1. Clone the repository:

   ```bash
   git clone (haev yet to decide)
   cd llm-factory
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys and configurations:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🚀 Quick Start

### Basic Example

```python
from llm_factory_updated import run_pipeline

# Simple text completion
result = run_pipeline(
    input_path_or_text="Explain quantum computing in simple terms.",
    client_type="azure",  # Options: "azure", "openai", "groq"
    pipeline_type="standard"  # Options: "standard", "cot" (Chain of Thought)
)

print(result)
```

### Chain of Thought Example

```python
# Multi-step reasoning
cot_config = {
    "name": "problem_solver",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "understand_problem",
            "prompt": "Analyze this math problem step by step: If x + y = 10 and x * y = 21, what are x and y?",
            "output_key": "problem_analysis"
        },
        {
            "type": "newQuestion",
            "name": "solve_problem",
            "prompt": "Now solve the problem using algebraic methods.",
            "input_key": "problem_analysis",
            "output_key": "solution"
        },
        {
            "type": "verification",
            "name": "verify_solution",
            "prompt": "Verify that your solution is correct by substituting the values back into the original equations.",
            "input_key": "solution",
            "output_key": "verified_solution"
        }
    ]
}

result = run_pipeline(
    prompt_config=cot_config,
    pipeline_type="cot"
)

print(result)
```

## 📁 Project Structure

```
project_root/
├── .env                     # Environment variables
├── env_loader.py            # Utility to load environment variables
├── llm_factory_updated.py   # Main factory implementation
├── example_usage.py         # Example usage
├── requirements.txt         # Dependencies
├── clients/                 # LLM client implementations
│   ├── __init__.py
│   ├── base_client.py       # Abstract base client
│   ├── azure_client.py      # Azure OpenAI implementation
│   ├── groq_client.py       # Groq implementation
│   └── openai_client.py     # OpenAI implementation
└── processors/              # Prompt processors
    ├── __init__.py
    ├── base_processor.py    # Abstract base processor
    ├── standard_processor.py # Standard extraction processor
    └── cot_processor.py     # Chain of Thought processor
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# Azure OpenAI Configuration
AZURE_BASE_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_API_KEY="your-azure-api-key-here"
AZURE_API_VERSION="2023-05-15"
AZURE_GPT_DEPLOYMENT_NAME="gpt-4-turbo"

# Groq Configuration
GROQ_API_KEY="your-groq-api-key-here"
GROQ_MODEL_NAME="llama3-70b-8192"

# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key-here"
OPENAI_MODEL_NAME="gpt-4o-mini"

# General LLM Settings
DEFAULT_TEMPERATURE=0
DEFAULT_MAX_TOKENS=8000

# Default Settings
DEFAULT_PIPELINE_TYPE="standard"
DEFAULT_CLIENT_TYPE="azure"
```

You only need to configure the providers you plan to use.

## 🧩 Usage Examples

### Standard Extraction with JSON Schema

```python
from llm_factory_updated import run_pipeline

# Define a JSON schema for structured output
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age", "occupation"]
}

# Create a prompt configuration
prompt_config = {
    "name": "person_extraction",
    "prompt": "Extract person details from the text into structured JSON.",
    "schema": person_schema,
    "context_data": "John Doe is a 35-year-old software engineer who is skilled in Python, JavaScript, and database design."
}

# Run the pipeline
result = run_pipeline(
    prompt_config=prompt_config,
    pipeline_type="standard"
)

print(result)
```

### Tonality Matching

```python
# Define tonality messages
tonality_messages = [
    {
        "role": "system",
        "content": "You are a professional medical writer. Format the extracted information in a clear, structured manner suitable for healthcare professionals."
    }
]

# Create a prompt configuration with tonality
prompt_config = {
    "name": "medication_extraction",
    "prompt": "Extract medication details from the text.",
    "schema": medication_schema,
    "tonality_messages": tonality_messages,
    "context_data": "Lisinopril 10mg tablets, take one tablet by mouth once daily for high blood pressure. Avoid taking with potassium supplements."
}

# Run the pipeline
result = run_pipeline(
    prompt_config=prompt_config,
    pipeline_type="standard"
)

print(result)
```

### Processing Multiple Extractions

```python
# Define multiple extraction configurations
extraction_configs = [
    {
        "name": "medication1",
        "prompt": "Extract medication details.",
        "schema": medication_schema,
        "context_data": "Lisinopril 10mg once daily"
    },
    {
        "name": "medication2",
        "prompt": "Extract medication details.",
        "schema": medication_schema,
        "context_data": "Metformin 500mg twice daily"
    }
]

# Run the pipeline with multiple extractions
results = run_pipeline(
    prompt_config=extraction_configs,
    pipeline_type="standard"
)

print(results)
```

### Custom Environment File

```python
# Use a custom environment file
result = run_pipeline(
    input_path_or_text="Analyze this text.",
    env_path="./config/production.env"
)
```

## 🔧 Advanced Features

### Chain of Thought Processing

Chain of Thought (CoT) processing allows you to create multi-step reasoning pipelines, where each step builds on the previous ones. This is particularly useful for complex reasoning tasks.

Available step types:

- `initialPrompt`: Sets up the reasoning pattern with examples
- `newQuestion`: Continues the reasoning with a new question
- `tonality`: Applies tonality formatting to previous outputs
- `followup`: Follow-up questions on previous steps
- `verification`: Verifies understanding or results
- `summary`: Summarizes the reasoning process

Example pipeline:

```python
cot_config = {
    "name": "medical_diagnosis_reasoning",
    "steps": [
        {
            "type": "initialPrompt",
            "name": "analyze_symptoms",
            "prompt": "Analyze the following symptoms: fever, headache, fatigue, and sore throat.",
            "output_key": "symptom_analysis"
        },
        {
            "type": "newQuestion",
            "name": "potential_diagnoses",
            "prompt": "Based on these symptoms, what are the potential diagnoses?",
            "input_key": "symptom_analysis",
            "output_key": "diagnoses"
        },
        {
            "type": "followup",
            "name": "additional_tests",
            "prompt": "What additional tests would you recommend to confirm the diagnosis?",
            "input_key": "diagnoses",
            "output_key": "recommended_tests"
        },
        {
            "type": "summary",
            "name": "treatment_plan",
            "prompt": "Summarize a potential treatment plan for the most likely diagnosis.",
            "input_keys": ["diagnoses", "recommended_tests"],
            "output_key": "treatment_plan"
        }
    ],
    "context_data": "Patient is a 45-year-old male with a history of seasonal allergies."
}
```

### JSON Schema Enforcement

You can enforce structured output using JSON schemas. This is particularly useful for extracting specific information in a consistent format.

Example JSON schema:

```python
medication_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "dosage": {"type": "string"},
        "frequency": {"type": "string"},
        "route": {"type": "string"},
        "indication": {"type": "string"},
        "side_effects": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "dosage", "frequency"]
}
```

## 🔌 Extending the Factory

### Adding New LLM Providers

1. Create a new client class in the `clients` directory:

```python
# clients/anthropic_client.py
from .base_client import BaseLLMClient

class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, api_key, model_name="claude-3-haiku", **kwargs):
        # Initialize Anthropic client
        self.api_key = api_key
        self.model_name = model_name
        # ...

    def generate_completion(self, messages, temperature=0, max_tokens=8000, **kwargs):
        # Implement completion generation
        # ...
```

2. Update the factory to include the new client:

```python
# llm_factory_updated.py
elif client_type.lower() == "anthropic":
    from clients.anthropic_client import AnthropicLLMClient
    return AnthropicLLMClient(**config_with_kwargs)
```

3. Add environment variables to `.env`:

```
# Anthropic Configuration
ANTHROPIC_API_KEY="your-anthropic-api-key"
ANTHROPIC_MODEL_NAME="claude-3-haiku"
```

### Creating Custom Processors

1. Create a new processor class in the `processors` directory:

```python
# processors/streaming_processor.py
from .base_processor import BasePromptProcessor

class StreamingProcessor(BasePromptProcessor):
    def process(self, client, prompt_config, **kwargs):
        # Implement streaming processing
        # ...
```

2. Update the factory to include the new processor:

```python
# llm_factory_updated.py
elif pipeline_type.lower() == "streaming":
    from processors.streaming_processor import StreamingProcessor
    return StreamingProcessor()
```

## ❓ Troubleshooting

### Common Issues

1. **"ImportError: No module named 'dotenv'"**
   - Solution: Install the python-dotenv package: `pip install python-dotenv`

2. **"KeyError: 'azure'"**
   - Solution: Check your `.env` file to ensure all required Azure variables are set

3. **"Failed to generate completion for 'X' after 3 attempts."**
   - Solution:
     - Check your API key and endpoint
     - Ensure your deployment exists and is running
     - Verify you haven't exceeded your rate limits

4. **"No schema matches the provided JSON"**
   - Solution: Check that your JSON schema definition matches the expected output structure

### Debug Mode

Enable debug logging by setting the environment variable:

```
LOG_LEVEL="DEBUG"
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests if available
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows the project's coding style and includes appropriate documentation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔎 What Can You Do With LLM Factory?

### 1. Build Conversational Applications

- Create chatbots with conversational memory
- Implement virtual assistants with multi-step reasoning
- Build customer support automation

### 2. Create Content Generation Tools

- Develop blog post generators with specific tonality
- Build product description writers with consistent formats
- Create document summarization tools

### 3. Implement Information Extraction Systems

- Extract structured data from unstructured text
- Build document parsing tools
- Create knowledge base population systems

### 4. Develop Decision Support Systems

- Implement diagnostic reasoning tools
- Create financial analysis assistants
- Build legal document analysis systems

### 5. A/B Test Different LLM Providers

- Compare quality across models with the same prompts
- Benchmark performance between providers
- Find the best cost-to-quality ratio for your use case

### 6. Create Educational Tools

- Build explanation systems with step-by-step reasoning
- Develop code generation and explanation tools
- Create interactive tutoring systems

### 7. Implement Complex Workflows

- Chain multiple extractions and generations together
- Build multi-agent systems
- Create systems that combine LLM outputs with other tools

---
