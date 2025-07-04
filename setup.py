from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llm_factory",
    version="0.1.3",
    description="A flexible framework for LLM interaction",
    author="LLM Factory Team",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['llm_factory', 'llm_factory.*']),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-core>=0.1.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "groq>=0.4.0",
        "ollama>=0.1.0",
        "pymupdf>=1.22.0",
        "python-docx>=0.8.11",
        "aspose.slides>=23.5.0",
        "pdf2image>=1.16.3",
        "Pillow>=9.5.0",
        "python-json-logger>=2.0.7",
        "requests>=2.31.0",
        "tqdm>=4.66.1",
    ],
    include_package_data=True,
)
