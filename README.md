# User-Manual-LLM
The developed system is an Information Retrieval QA system that answers based on the document it is fed. It is powered by three main pipelines (Figure 1):

![flowchart](https://github.com/user-attachments/assets/6a1ebbc1-fdf6-4f49-94ed-1c222d821109)

## The Data Ingestion Pipeline

It is responsible for the consumption and processing of the user manual uploaded by the user. This pipeline is critical to the performance of the whole system based on the fundamental principle of computing, “Garbage-In Garbage-Out”. 

The first step in this pipeline is _**text extraction**_. Once the user manual is uploaded by the user in PDF format, the text in the manual is extracted. The extraction is tricky because different user manuals have different formatting (e.g., multiple columns, presence of headers and footers, images, tables, etc.) while most PDF text extraction libraries take contents only in a straight line without respect for reading order. To get around the formatting issue, we used Marker-pdf (https://github.com/VikParuchuri/marker), a library built on a combination of deep learning models that can convert a wide range of PDFs to markdown, HTML or JSON formats. Using the library, we get a text variable holding the texts in the user manual in Markdown format with references to locations where images were and tables from the uploaded PDF, all the images stored in a list and a dictionary containing metadata, including a table of contents. 

Next, _**we use Python regular expressions to filter out all symbols, characters, and information that would not contribute positively to the system's performance**_. We also filter out texts that are not in English using the googletrans 4.0.0-rc1 library. Knowing that most open-source language models have a context window of ~512 tokens, _**we broke the texts into chunks and organised them using the names of the sections they came from**_. Finally, we use the multi-qa-mpnet-base-dot-v1 sentence transformer (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) to _**convert all the chunks into their dense vector representations**_ before _**storing the representations in a Facebook AI Similarity Search (FAISS) vector store**_.

## The Context-Retrieval Pipeline

This pipeline acts like a sieve. It only gives the QA model chunks related to the query, which we refer to as context from here on. This pipeline starts with the user asking a question in Pidgin. The question is converted to English using the NITHUB-AI/marian-mt-bbc pcm-en translation model (https://huggingface.co/NITHUB-AI/marian-mt-bbc-en-pcm). The dense vector representation of the translated question is generated using the multi-qa-mpnet-base-dot-v1 sentence transformer (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1). To retrieve the context related to the query, a K-nearest neighbour search based on the Euclidean distance is carried out between the embedded query and all the vectors in the FAISS vector DB. To that end, the top 3 chunks with the smallest Euclidean Distances are restructured, placed in a paragraph and sent to the QA model as the context for the question.

## The Multilingual Question Answering Pipeline

In this pipeline, the question is answered using Google's FLAN-T5-large model, and the answer is translated into Pidgin. The QA model takes two things as input: the translated query from the user and the context from the Context-Retrieval pipeline. The answer from the QA model is fed to the NITHUB-AI/marian mt-bbc-en-pcm translation model (https://huggingface.co/NITHUB-AI/marian-mt-bbc-en-pcm) – it is the same model used to translate the query to English, but it is translating from English to Pidgin in this scenario. The Pidgin answer is then returned to the user. 

## Evaluation

The system's overall computation time when responding to a query without a GPU was measured and found to be between 4 and 7 minutes. However, no other outputs were measured as all the models used in the system are already benchmarked and have yet to be fine-tuned.
