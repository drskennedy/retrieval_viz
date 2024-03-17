# Visualizing Naive and Advanced Retrievers' Impact Using Spotlight

**Step-by-step guide on Medium**: [Understanding Impact of Advanced Retrievers  on RAG Behavior through Visualization](https://medium.com/@heelara/understanding-impact-of-advanced-retrievers-on-rag-behavior-through-visualization-e7670804fd05)
___
## Context
Retrieval-Augmented Generation (RAG) is a popular technique used to improve the text generation capability of an LLM by keeping it fact driven and reduce its hallucinations. RAG performance is directly influenced by the embeddings formed from the chosen documents.
In this project, we will develop a RAG application using FAISS vectorstore and TinyLlama 1.1B Chat. The design of the app is per below:
<br><br>
![System Design](/assets/retrieval_viz_design.png)

To understand the impact of a document retrieval on RAG behavior, visualization library `renumics-spotlight` is used.
<br><br>
![UMAP vs PCA Visualizations](/assets/UMAPvsPCA.png)
___
## How to Install
- Create and activate the environment:
```
$ python3.10 -m venv mychat
$ source mychat/bin/activate
$ cd mychat
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf from [TheBloke HF report](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) to sub-directory `./models/tinyllama/`.
- Run script `main.py` to start the app:
```
$ python main.py
```
___
## Quickstart
- To start the app, launch terminal from the project directory and run the following command:
```
$ source mychat/bin/activate; cd mychat
$ python main.py
```
- Here is a sample run:
```
$ python main.py
Q: What is a major issue with policy-based routing virtual in-path SteelHead deployment?
A: A major issue with policy-based routing (PBR) network deployment for virtual in-path SteelHeads is that it can cause a traffic black hole if the PBR next-hop IP address is unavailable. This issue can occur if the PBR next-hop IP address is not available due to a failure of the SteelHead or due to a network partition or other issues. To avoid this issue, you can configure the PBR-enabled router to use the CiSCo Discovery Protocol (CDP) to obtain information about neighbor IP addresses and models. This protocol enables the PBR router to obtain information about neighbor IP addresses and models even if the PBR next-hop IP address is unavailable.
```
Here is a screenshot of the visualization from this run:
<br><br>
![Vector Space Visualization](/assets/ui_screenshot.png)
___
## Key Libraries
- **LangChain**: The framework library for developing applications powered by language models.
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Spotlight**: Visualization library to interactively explore unstructured datasets.

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `opdf_index`: Directory for FAISS index and vectorstore
- `main.py`: Main Python module to launch the application
- `LoadVectorize.py`: Python module to load a pdf document, split and vectorize
- `requirements.txt`: List of Python dependencies (and versions)
___

## References
- https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
