# --------------Installing googletrans 4.0.0-rc1-----------------------
!git clone https://github.com/ssut/py-googletrans
%cd /content/py-googletrans/googletrans
!git checkout 934981f

# ---------------Addressing version incompatibility----------------------
!pip install httpx==0.28.0

# Path to the googletrans client.py file
file_path = '/content/py-googletrans/googletrans/client.py'

# Open the file and read its content
with open(file_path, 'r') as file:
    file_content = file.readlines()

# Modify the content of the file where the 'httpx.Client' is instantiated
for i, line in enumerate(file_content):
    if "self.client = httpx.Client(" in line:
        # Replace the line with a modified version without the proxies argument
        file_content[i] = line.replace(", proxies=proxies", "")

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.writelines(file_content)

print("File updated successfully.")

!pip install httpx[http2]

# ----------------Required Library Installations---------------------
%pip install gradio
%pip install marker-pdf==1.0.0
%pip install langchain
%pip install transformers
%pip install sentence-transformers
%pip install faiss-cpu
%pip install numpy

# ----------------Main Framework Code----------------------------------
import sys
sys.path.append('/content/py-googletrans')  # Adding the cloned directory to the path

import gradio as gr
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# Initialize Models (global)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
qa_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
qa_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
translation_tokenizer = AutoTokenizer.from_pretrained("NITHUB-AI/marian-mt-bbc-en-pcm")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("NITHUB-AI/marian-mt-bbc-en-pcm")
#translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-lv")
#translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-lv")
#translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
#translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

# Globals for FAISS Index and Chunks
faiss_index = None
chunked_data = []

# Initialize models and configurations
def initialize_converter():
    config = {"output_format": "markdown", "language": "en"}
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    return converter

def extract_metadata_and_text(converter, pdf_file):
    rendered = converter(pdf_file)
    full_text, _, images = text_from_rendered(rendered)
    metadata = rendered.metadata

    return metadata, full_text, images

def filter_titles(titles, translator):
    print("\nFiltering titles...")
    cleaned_titles = []
    for title in titles:
        try:
            detection = translator.detect(title)
            if detection.lang == 'en':
                cleaned_titles.append(title.strip())
        except Exception as e:
            continue

    return cleaned_titles

def filter_markdown_by_language(full_text, english_titles, translator):
    print("\nFiltering texts...")
    lines = full_text.split("\n")
    filtered_text = []
    retain_section = False

    for line in lines:
        line = line.strip()

        if line.startswith("#"):
            current_section = line
            retain_section = any(title in current_section for title in english_titles)
            if retain_section:
                filtered_text.append(line)
        elif retain_section:
            try:
                detection = translator.detect(line)
                if detection.lang == "en":
                    filtered_text.append(line)
            except Exception as e:
                continue

    return "\n".join(filtered_text)

def chunk_sections(filtered_text, splitter, cleaned_titles):
    print("\nCreating chunks from the PDF...")

    text_pattern = r'([#*]+)|-\s'

    cleaned_text = re.sub(text_pattern, '', filtered_text)

    # Step 1: Escape headers for regex
    escaped_headers = [re.escape(header) for header in cleaned_titles]
    #print("Escaped Headers:", escaped_headers)

    # Step 2: Build regex pattern to match headers and keep whitespace
    pattern = r'(?=\n|^)(\s*(' + '|'.join(escaped_headers) + r')\s*\n)(.*?)(?=\n\s*(?:' + '|'.join(escaped_headers) + r')|$)'

    # Step 3: Use re.findall to capture the headers and content as separate groups
    matches = re.findall(pattern, cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    #print("Matches found by regex:", matches)

    if not matches:
        print("No matches found. Check your regex pattern or filtered_text.")
        return []

    # Step 4: Process the matches and join the content properly
    sections = {}
    for match in matches:
        header = match[1].strip()  # Extract the header
        content = match[2].strip()  # Extract the content and remove unwanted

        # Find image patterns and remove them from content
        image_pattern = r'!\[\]\(([^)]+)\)'  # Regex to match image patterns
        found_images = re.findall(image_pattern, content)  # Extract image patterns
        content = re.sub(image_pattern, '', content)  # Remove image patterns from content

        # Make the content a single line by replacing internal newlines with spaces
        content = " ".join(content.split())

        # If the header already exists, concatenate the new content and images
        if content:
            if header in sections:
                sections[header]['content'] += " " + content
                sections[header]['images'].extend(found_images)
            else:
                sections[header] = {'content': content, 'images': found_images}

    # Print the sections with header and content
    #for s_header, s_content in sections.items():
        #print(f"Header: {s_header}")
        #print(f"Content: {s_content['content'][:500]}...")  # Print first 500 chars for brevity

    # Step 5: Chunk the sections
    chunked_data = []
    for section_header, section_content in sections.items():
        content = section_content['content']
        images = section_content['images']

        if len(content) > 500:
            chunks = long_section_chunks(content, splitter)
        else:
            chunks = short_section_chunks(content, splitter)

        #print(f"Chunks created for section '{section_header}': {chunks}")

        for i, chunk in enumerate(chunks):
            chunk_header = f"{section_header} {i + 1}" if len(chunks) > 1 else section_header
            chunked_data.append({
                'header': chunk_header,
                'content': f"{section_header}: {chunk}",
                'images': images
            })

    #print("Final Chunked Data:", chunked_data)
    return chunked_data

# Clean incomplete phrases at the ends of chunks
def clean_incomplete_start(chunks):
  """
  Removes incomplete phrases at the beginning of each chunk.
  """
  cleaned_chunks = []
  for i, chunk in enumerate(chunks):
      # Only process chunks after the first one to clean the beginning
      if i > 0:
          # Look for the first complete sentence
          match = re.search(r"[A-Z][^.!?]*[.!?]", chunk)
          if match:
              chunk = match.group(0).strip() + chunk[match.end():]  # Keep the full text after the first sentence
      cleaned_chunks.append(chunk)
  return cleaned_chunks

# Helper functions for chunking
def long_section_chunks(long_content, splitter):
  long_chunks = splitter.split_text(long_content)

  long_chunks = clean_incomplete_start(long_chunks)

  return long_chunks

def short_section_chunks(short_content, splitter):
    return splitter.split_text(short_content)

def generate_embeddings(chunked_data, model):
    print("\nGenerating embeddings...")
    for chunk in chunked_data:
        embedding = model.encode(chunk['content'])
        chunk['embedding'] = embedding
    return chunked_data

def faiss_index_creation(chunked_data):
    print("\nStoring the PDF chunks in VectorStore...")
    embedding_matrix = np.array([chunk["embedding"] for chunk in chunked_data])
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index

def query_faiss(query, model, index, chunked_data):
    query_embedding = model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), k=4)
    retrieved_chunks = [chunked_data[i] for i in indices[0]]
    return retrieved_chunks

def structured_context(retrieved_chunks, chunked_data):
    retrieved_headers = set(chunk["header"] for chunk in retrieved_chunks)
    section_data = {}

    for chunk in chunked_data:
        if chunk["header"] in retrieved_headers:
            base_header = re.sub(r"\s\d+$", "", chunk["header"])
            if base_header not in section_data:
                section_data[base_header] = []
            section_data[base_header].append(chunk["content"])

    final_context = ""
    for header, contents in section_data.items():
        combined_content = " ".join(contents)
        content_without_header = combined_content.replace(f"{header}:", "").strip()
        final_context += f"{header}:\n{content_without_header}\n\n"

    return final_context.strip()

# Gradio Interface Functions
def process_pdf(pdf_file, query):
    global faiss_index, chunked_data

    try:
      # Initialize the converter
        yield "Step 1/6: Initializing converter...", gr.update(value=15, visible=True)
        converter = initialize_converter()
        time.sleep(0.5)  # Simulate processing time for feedback

        # Extract metadata and text from PDF
        yield "Step 2/6: Extracting metadata and text...", gr.update(value=30, visible=True)
        metadata, full_text, images = extract_metadata_and_text(converter, pdf_file.name)  # Fix the file handling
        titles = [entry["title"].replace("\n", "") for entry in metadata["table_of_contents"]]
        time.sleep(0.5)  # Simulate processing time for feedback

        # Initialize translator and filter titles
        yield "Step 3/6: Filtering titles and content...", gr.update(value=45, visible=True)
        translator = Translator()
        cleaned_titles = filter_titles(titles, translator)

        # Filter text by language
        filtered_text = filter_markdown_by_language(full_text, cleaned_titles, translator)
        time.sleep(0.5)  # Simulate processing time for feedback

        # Chunking and processing
        yield "Step 4/6: Breaking the sections into smaller parts...", gr.update(value=60, visible=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        chunked_data = chunk_sections(filtered_text, splitter, cleaned_titles)
        time.sleep(0.5)  # Simulate processing time for feedback

        # Generate embeddings
        yield "Step 5/6: Teaching the computer your PDF...", gr.update(value=80, visible=True)
        chunked_data = generate_embeddings(chunked_data, model)
        time.sleep(0.5)  # Simulate processing time for feedback

        # Create FAISS index
        yield "Step 6/6: Storing the knowledge in the computer's brain...", gr.update(value=100, visible=True)
        faiss_index = faiss_index_creation(chunked_data)
        time.sleep(0.5)  # Simulate processing time for feedback

        yield "Processing complete! You can now ask questions.", gr.update(value=100, visible=True)

    except Exception as e:
        yield f"Error during processing: {str(e)}", gr.update(value=0, visible=True)

# Function to handle query processing
def process_query(query):
    global faiss_index, chunked_data

    if not faiss_index or not chunked_data:
        return "No PDF processed yet. Please upload and process a PDF.", ""

    # Query FAISS index
    retrieved_chunks = query_faiss(query, model, faiss_index, chunked_data)

    # Generate final context
    context = structured_context(retrieved_chunks, chunked_data)
    print(context)

    # QA with FLAN-T5 model
    #qa_input = f"Think step by step to answer the question based on the context.\n\nContext: {context}\n\nQuestion: {query}"

    qa_input = (
    f"Carefully analyze the context and provide the most relevant and concise answer to the question. "
    f"If the answer requires a specific number, name, or brief fact, extract and present it directly. "
    f"If the answer requires a detailed explanation, summarize it clearly and logically without repeating the entire context. "
    f"Avoid including unnecessary information.\n\n"
    f"Context: {context}\n\n"
    f"Question: {query}\n\n"
    f"Answer:"
)
    inputs = qa_tokenizer(qa_input, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = qa_model.generate(
            inputs["input_ids"],
            max_length=500,
            num_beams=5,
            temperature=0.3,
            top_p=0.8,
            early_stopping=False
        )

    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Translate the answer to Pidgin English
    translated_input = translation_tokenizer(answer, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        translated_output = translation_model.generate(
            translated_input["input_ids"],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    translated_answer = translation_tokenizer.decode(translated_output[0], skip_special_tokens=True)

    #return translated_answer
    return answer, translated_answer

# Function that tracks the progress and enables the query input after step 6/6
def after_processing_pdf():
    # When processing is at step 6, enable the query input field
    return gr.update(interactive=True)

# Assuming the `progress_slider` is updated within your processing steps, you can call this at the end (step 6/6).
def update_progress(value):
    # Check if the progress is at step 6/6 (you may need to adjust this based on your implementation)
    if value == 100:  # If the progress reaches 100 (step 6/6)
        return after_processing_pdf()  # Enable query input box
    else:
        return gr.update(interactive=False)  # Keep the input disabled until step 6/6

# Gradio UI setup with Blocks
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", type="filepath")
            upload_button = gr.Button("Upload")
            progress_slider = gr.Slider(
                label="Processing Progress",
                value=0,
                minimum=0,
                maximum=100,
                interactive=False,
                visible=False,
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Processing status will appear here."
            )

    with gr.Row():
        query_input = gr.Textbox(
            label="Enter Query",
            placeholder="Ask a question...",
            interactive=False,  # Disabled initially
        )
        submit_button = gr.Button("Enter", interactive=False)  # Disabled initially

    with gr.Row():
        original_answer = gr.Textbox(label="Original Answer")
        pidgin_answer = gr.Textbox(label="Answer in Pidgin English")
        #pidgin_answer = gr.Textbox(label="Answer in Dutch")

    # Link Upload button to backend processing
    upload_button.click(
        process_pdf,  # Backend function
        inputs=[pdf_input],
        outputs=[status_output, progress_slider],  # Updates slider and status text
    )

    # Update query_input only after step 6/6
    def enable_query_input(progress_value):
        if progress_value == 100:  # Step 6/6 completion
            return gr.update(interactive=True)
        return gr.update(interactive=False)

    progress_slider.change(
        enable_query_input,  # Check if the progress is 100 (step 6/6)
        inputs=[progress_slider],
        outputs=[query_input]  # Enable the query input after processing is done
    )

    # Enable/Disable submit button based on the input query
    query_input.change(
        lambda query: gr.update(interactive=True) if query.strip() != "" else gr.update(interactive=False),
        inputs=[query_input],
        outputs=[submit_button]
    )

    # Submit button for querying
    submit_button.click(
        process_query,
        inputs=[query_input],
        outputs=[original_answer, pidgin_answer],
        #outputs=[pidgin_answer],
    )

# Launch the Gradio interface
interface.launch(debug=True)
