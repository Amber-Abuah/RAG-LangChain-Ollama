## Retrieval Augmented Generation (RAG) With LangChain And Ollama

Embeddings: [nomic-embed-text](https://ollama.com/library/nomic-embed-text)  
Model: [Llama3](https://ollama.com/library/llama3)  
Context Document: [Cat Wikipedia Page](https://en.wikipedia.org/wiki/Cat) converted to a PDF.

A Python app where users are able to converse with a local Large Language Model (in this case Llama3 through [Ollama](https://ollama.com/)) about their local own documents.  

The embeddings, model and context documents are easily interchangeable; update the `vector_database embedding` parameter, `model_name` and `pdf_path` variables accordingly.

### Process:
- The document is loaded and split into smaller chunks.
- These chunks are then stored in a Chroma vector database for retrieval.
- When the user enters a prompt, context is retrieved from the Chroma database and appended onto the user's initial prompt as context.
- The LLM then produces a response to the full prompt, using the context to base its answer off of.

### An example exchange between the user and Llama3:
```
>> Summarise the document

The document provides information about a cat's physical characteristics and senses. Here are the main points:
* Eyes:
        + Have a tapetum lucidum that reflects light back into the eye, increasing sensitivity to dim light
        + Pupils expand in low light levels to cover most of the exposed surface
        + Color vision is limited, with only two types of cone cells sensitive to blue and yellowish green
* Hearing:
        + Most acute in the range of 500 Hz to 32 kHz
        + Can detect a broad range of frequencies (55 Hz to 79 kHz) and ultrasound
        + Has large movable outer ears that amplify sounds and help detect location of noise
* Smell:
        + Has an acute sense of smell due to a well-developed olfactory bulb and large surface of olfactory mucosa
        + Can sense certain aromas through a Jacobson's organ in the mouth
        + Responds strongly to pheromones, including those found in catnip
* Taste:
        + Has relatively few taste buds compared to humans
        + Possesses receptors for acids, amino acids, and bitter tastes
        + Can detect umami flavors, particularly in tuna, due to specific combination of molecules

Overall, the document highlights a cat's unique physical characteristics and senses that help them navigate their environment.
```

Inspired by [tonykipkemboi's ollama_pdf_rag](https://github.com/tonykipkemboi/ollama_pdf_rag) tutorial and repository.