# FAQ Retrieval Assistant

A lightweight FAQ retrieval assistant built using OpenAI embeddings, designed to support both **English and Macedonian queries**, with confidence-aware responses and a minimal web interface.

---

## Approach

### 1. Planning and baseline implementation

I started by breaking the task into smaller, testable milestones.  
The first goal was to build a **minimal but complete baseline system**:

- Generate a dataset of **20 FAQ question–answer pairs**
- Embed all questions using a **multilingual OpenAI embedding model**
- Implement a semantic retrieval pipeline based on **cosine similarity**
- Expose the system through a **minimal Flask web interface**
- Add a **confidence score** to express how reliable the retrieved results are

At this stage, the system supported semantic retrieval but treated each question independently.

---

### 2. Confidence scoring and UX thresholding

To avoid blindly returning results for ambiguous queries, I introduced a **confidence score** based on:

- The similarity score of the top retrieved result
- The **margin** between the top result and the second-best result

The idea is simple:
- High similarity + clear margin → confident answer
- Close similarities → ambiguous intent

This confidence score is compared against a configurable **threshold**.  
If the confidence is below the threshold, the UI transparently informs the user:

> *“I’m not fully confident, but these might help…”*

This improves UX by making uncertainty explicit rather than hiding it.

---

### 3. Evaluation-driven development (baseline)

To avoid tuning the system intuitively, I created an **evaluation dataset** consisting of:
- English queries
- Macedonian queries
- Ambiguous and noisy formulations

Using Jupyter notebooks, I evaluated:
- Top-1 accuracy
- Top-3 accuracy
- Confidence behavior
- Language-specific performance

This baseline evaluation showed that **Macedonian queries performed poorly**, which was expected since only English questions were embedded.

---

### 4. Enhancing multilingual performance

Improving Macedonian performance required careful design decisions focused on performance and simplicity.

I considered multiple approaches:
- Switching to a different embedding model
- Translating queries at runtime
- Language classification with routing to multiple indexes

I ultimately chose a solution that balances **performance, correctness, and code simplicity**:

**Multilingual question variants with canonical answers**

- Each FAQ answer has:
  - One canonical **English answer**
  - Multiple question variants (English + Macedonian)
- All question variants are embedded
- Variants map to the same `answer_id`
- Retrieval is performed over all questions
- Results are **deduplicated by answer**
- The UI always displays the **English canonical question**

This approach avoids runtime translation, additional indexes, and unnecessary latency, while significantly improving Macedonian query recall.

---

### 5. Re-evaluation and threshold re-tuning

After adding Macedonian question variants:
- I re-ran the full evaluation using new notebooks
- Macedonian Top-1 and Top-3 accuracy improved significantly
- Confidence distributions shifted upward

Based on these results, the confidence threshold was **re-estimated empirically**, resulting in a stricter and more reliable threshold.

---

### 6. Refactoring, robustness, and configurability

Once functionality was complete, I refactored the codebase with a focus on:

- Readability and separation of concerns
- Explicit error handling and graceful failure
- Clear data-loading boundaries
- Configuration through a dedicated `config.py`

The following parameters are configurable:
- OpenAI embedding model
- Number of top results (`TOP_K`)
- Confidence threshold

Meaningful error messages are propagated to the UI without crashing the application.

---

### 7. Dockerization

Docker configuration was added to allow the project to be run without any local Python setup.

The container:
- Generates embeddings on startup
- Runs the Flask application
- Exposes the app on a fixed port

This enables quick setup and easy review.

---

## Tools Used

- **OpenAI API** – multilingual text embeddings
- **Flask** – lightweight web framework
- **NumPy** – vector operations and similarity computation
- **Pandas** – evaluation and analysis
- **Matplotlib** – visualization during evaluation
- **Docker / Docker Compose** – containerized execution

---

## How to Run the Project

### Prerequisites
- Docker
- Docker Compose

---

### 1. Set up environment variables

Create a `.env` file based on `.env.example` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```
### 2. Build and run the application

From the project root, build and start the application using Docker Compose:

```bash
docker-compose up --build
```

On startup:
- FAQ embeddings are generated automatically
- The Flask application is launched inside the container

---

### 3. Use the application

Once the container is running, open your browser at:

```
http://localhost:5001
```

You will see a minimal HTML interface where you can:
- Enter queries in English or Macedonian
- View the top retrieved FAQ answers
- See a confidence-aware response indicating how reliable the retrieval is

If the confidence score is below the configured threshold, the UI will explicitly indicate uncertainty instead of returning a misleading answer.

---

### 4. Inspect and configure the system

You can inspect the FAQ dataset in:

```
data/faqs.json
```

This file contains the multilingual question variants used for retrieval.

Canonical answers are defined in:

```
data/answers.json
```

System behavior can be configured via:

```
app/config.py
```

After changing any configuration, rebuild the container for changes to take effect.

---

### 5. Embeddings and data notes

- The generated embeddings are stored in:

```
data/embeddings.npy
```

This file is not tracked in version control and can be regenerated at any time
- Embeddings are deterministic for a given model and input
- Regeneration is handled automatically when the container starts

---

## 6. Evaluation artifacts

The repository includes evaluation notebooks used to:
- Measure baseline retrieval performance
- Analyze confidence–coverage tradeoffs
- Compare baseline and enhanced (multilingual) retrieval quality

Evaluation results are stored under:

```
eval/results/
```

In addition to the individual evaluation notebooks, a dedicated
`comparison_evaluation.ipynb` notebook directly compares the baseline
and enhanced systems, clearly highlighting the accuracy improvements
achieved by the multilingual approach, particularly for Macedonian queries.

These artifacts demonstrate measurable improvements and support
data-driven threshold selection rather than intuitive tuning.
---

## 7. Stopping the application

To stop the running containers, use:

```bash
docker-compose down
```

---

## 8. Summary

This project demonstrates:
- A clean semantic retrieval pipeline using embeddings
- Multilingual query support without runtime translation
- Confidence-aware UX behavior
- Evaluation-driven iteration
- Robust, containerized execution

The focus is on clarity, correctness, and measurable improvement rather than unnecessary complexity.
