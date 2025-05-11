import json
import re
from datetime import datetime
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean the text by removing unwanted patterns and normalizing whitespace"""
    # Only remove page numbers pattern (e.g., "Page X of Y")
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Remove copyright line
    text = re.sub(r'© Xebia 2023 Confidential: Not for Distribution\. www\.xebia\.com', '', text)
    
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up dots pattern
    text = re.sub(r'\.{3,}', '...', text)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_page_content(page: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and combine text, image, and table content from a page in reading order"""
    all_content = []
    table_positions = set()  # Track table positions
    
    # First pass: Identify table positions to avoid duplicate text
    for content_item in page.get('content', []):
        if 'table_text' in content_item:
            y1 = content_item.get('y1', 0)
            y2 = content_item.get('y2', 0)
            # Add some buffer around table position
            table_positions.add((y1 - 5, y2 + 5))
    
    # Second pass: Process content
    for content_item in page.get('content', []):
        item_type = content_item.get('type')
        position = content_item.get('position', {})
        
        # Skip text that overlaps with table positions
        if item_type == 'text':
            text_y = position.get('y1', 0)
            skip_text = any(y1 <= text_y <= y2 for y1, y2 in table_positions)
            if skip_text:
                continue
            
            text = clean_text(content_item.get('content', ''))
            if text:
                all_content.append({
                    'text': text,
                    'position': position,
                    'type': 'text'
                })
                
        elif item_type == 'image' and content_item.get('extracted_text'):
            text = content_item.get('extracted_text', '').strip()
            if text:
                all_content.append({
                    'text': text,
                    'position': position,
                    'type': 'image'
                })
                
        elif 'table_text' in content_item:
            table_text = content_item.get('table_text', '')
            if table_text:
                # Extract and format table content
                table_content = re.search(r'##.*?##\s*(.*?)\s*##.*?##', table_text, re.DOTALL)
                if table_content:
                    table_rows = table_content.group(1).strip().split('\n')
                    # Remove the first row if it's just numbers (0|1|2|3)
                    if table_rows and re.match(r'^[\d\|\s]+$', table_rows[0]):
                        table_rows = table_rows[1:]
                    formatted_table = '\n' + '\n'.join(table_rows) + '\n'
                    
                    table_position = {
                        'x1': content_item.get('x1'),
                        'y1': content_item.get('y1'),
                        'x2': content_item.get('x2'),
                        'y2': content_item.get('y2')
                    }
                    all_content.append({
                        'text': formatted_table,
                        'position': table_position,
                        'type': 'table'
                    })
    
    # Sort content by vertical position then horizontal position
    sorted_content = sorted(all_content, 
                          key=lambda x: (x['position'].get('y1', 0), x['position'].get('x1', 0)))
    
    def clean_table_formatting(text: str) -> str:
        """Clean up table formatting while preserving structure"""
        # Remove empty cells
        text = re.sub(r'\|\s*\|', '|', text)
        # Remove leading/trailing pipes
        text = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', text, flags=re.MULTILINE)
        # Normalize spaces around pipes
        text = re.sub(r'\s*\|\s*', ' | ', text)
        return text.strip()

    def clean_non_table_text(text: str) -> str:
        """Clean up non-table text"""
        # Remove any stray pipes
        text = re.sub(r'\s*\|\s*', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Combine content
    combined_text = []
    for item in sorted_content:
        if not item['text'].strip():
            continue
            
        if item['type'] == 'table':
            # Clean up table formatting
            cleaned_text = clean_table_formatting(item['text'])
            if cleaned_text:
                combined_text.append(cleaned_text)
        else:
            # Clean up regular text
            cleaned_text = clean_non_table_text(item['text'])
            if cleaned_text:
                combined_text.append(cleaned_text)
    
    # Join all content with proper spacing
    final_text = ' '.join(combined_text)
    
    # Final cleanup
    final_text = re.sub(r'\n\s+', '\n', final_text)  # Remove extra spaces after newlines
    final_text = re.sub(r'\s+\n', '\n', final_text)  # Remove extra spaces before newlines
    final_text = re.sub(r'\s+', ' ', final_text)     # Normalize spaces
    
    return {
        'text': final_text.strip(),
        'page_number': page['page_number']
    }

def sort_chunks_by_page(json_content):
    """
    Sort document chunks by page number. Modifies json_content in place.
    
    Args:
        json_content (dict): Document JSON containing 'chunks' key
    
    Returns:
        dict: The same JSON with sorted chunks
    """
    if 'chunks' in json_content:
        json_content['chunks'] = sorted(
            json_content['chunks'],
            key=lambda x: x.get('page_number', 0) if isinstance(x, dict) else 0
        )
    return json_content

def context_aware_chunking_strategy(chunking_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process document content using context-aware chunking strategy"""
    structured_page_info = []
    options = chunking_config['chunking_options']
    pages = chunking_config['docs']
    
    if 'page_level' in options:
        for page in pages:
            page_content = extract_page_content(page)
            if page_content['text'].strip():
                # Clean up any multiple newlines or spaces
                cleaned_text = re.sub(r'\n{3,}', '\n\n', page_content['text'])
                cleaned_text = re.sub(r' {3,}', '  ', cleaned_text)
                
                structured_page_info.append({
                    "text": cleaned_text,
                    "page_number": page_content['page_number'],
                    "type": "page_level"
                })
    
    return structured_page_info

def process_document_chunks(json_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process document content and create chunks.
    Return the full processed object including chunks.
    """
    chunking_config = {
        "chunking_options": [
            "page_level",
            "tables"
        ]
    }
    print("Chunking...")
    
    # Extract pages from the input
    if isinstance(json_content, list):
        pages = json_content
    elif isinstance(json_content, dict):
        pages = json_content.get('pages', [])
        if not pages and 'documents' in json_content:
            for doc in json_content['documents']:
                if 'document_content' in doc:
                    doc_content = json.loads(doc['document_content'])
                    pages = doc_content.get('pages', [])
                    break
    
    chunking_config["docs"] = pages
    
    # Generate chunks
    chunks = context_aware_chunking_strategy(chunking_config)
    
    # Modify the input json_content to include chunks
    if isinstance(json_content, dict):
        json_content['chunks'] = chunks
    else:
        json_content = {
            'pages': pages,
            'chunks': chunks
        }
        
    json_content = sort_chunks_by_page(json_content)

    return json_content

def save_processed_document(processed_document: Dict[str, Any], filename: str = None) -> str:
    """Save processed document with chunks to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_document_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_document, f, indent=2, ensure_ascii=False)
    
    return filename