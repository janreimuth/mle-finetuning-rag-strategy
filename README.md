# Fine-Tuning and RAG Strategy

This repository teaches how to choose between
Retrieval-Augmented Generation (RAG),
fine-tuning, and hybrid architectures for domain-specific LLM systems. It uses
real chat-completion calls through the
OpenAI Python client, with
Groq OpenAI compatibility as the
primary provider and Mistral chat completions
as an alternative provider.

## Learning Objectives

By the end of this repository, you should be able to:

- Compare RAG-only, fine-tuning-style, and hybrid architectures on a domain task.
- Explain which failure modes each strategy can and cannot address.
- Inspect whether a dataset is suitable for retrieval, supervised fine-tuning, or evaluation.
- Build provider-backed baselines that make strategy trade-offs measurable without a paid training job.
- Justify a strategy recommendation with measured results and criteria such as freshness, auditability, and cost.

## Scope

This repository does not train a model. Actual fine-tuning, with LoRA adapters
via Hugging Face PEFT and a hand-written PyTorch training loop, is covered in the
preceding **Fine-Tuning Small Models Locally + Evaluation** unit, and this
repository assumes you know those mechanics. Here the question is architectural:
when should a team use RAG, fine-tuning, or a hybrid path? Notebook 04 uses
few-shot examples as a lightweight stand-in for fine-tuning, and notebook 05
compares RAG-only, behavior-adapted, and hybrid prompts on the same domain task.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - RAG vs Fine-Tuning Strategy Map**](01-rag-vs-finetuning-strategy-map.ipynb) | Compare parametric knowledge, retrieved evidence, and behavior adaptation. |
| [**02 - Domain Data and Fine-Tuning Readiness**](02-domain-data-and-finetuning-readiness.ipynb) | Inspect retrieval, behavior-example, and evaluation data contracts. |
| [**03 - RAG-Only Baseline and Failure Modes**](03-rag-only-baseline-and-failure-modes.ipynb) | Build a transparent retriever and inspect cited answers and abstention. |
| [**04 - Fine-Tuning-Style Baseline and Limits**](04-finetuning-only-baseline-and-limits.ipynb) | Use few-shot style conditioning as a proxy, not a real fine-tuning run. |
| [**05 - Hybrid Architecture Decision Report**](05-hybrid-architecture-decision-report.ipynb) | Compare strategy signals and justify the next experiment. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**assets**](assets/) | Visual aids referenced in the repository. |
| [**data**](data/) | Policy documents, support requests, and style examples. |
| [**strategy_helpers.py**](strategy_helpers.py) | Shared data, provider, and sparse retrieval helpers. |
| [**test_notebook_contracts.py**](test_notebook_contracts.py) | Offline regression checks for maintainers. |
| [**.env.example**](.env.example) | Optional provider configuration template. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 or newer
and Python 3.13.17 or a newer 3.13 patch release, as defined in
[`.python-version`](.python-version).

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-finetuning-rag-strategy`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

The command below installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

All five notebooks run locally without a model download or API key. Without credentials, provider outputs are explicitly simulated examples, not measurements of model quality. Each reply carries a `live`, `simulated`, or `fallback` source label.

---

### 5. Set Up Environment Variables (Optional)

For live model calls, copy the example file and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set `LLM_PROVIDER` to `groq` or `mistral`, then add the matching key:

- **Groq (primary):** Create a key at [console.groq.com/keys](https://console.groq.com/keys) and set `GROQ_API_KEY`.
- **Mistral (alternative):** Create a key following the [Mistral API key documentation](https://docs.mistral.ai/admin/security-access/api-keys) and set `MISTRAL_API_KEY`.

`LLM_PROVIDER` selects the preferred provider. If its key is missing, the helper selects the other configured provider. `GROQ_MODEL` and `MISTRAL_MODEL` override model names.

> [!CAUTION]
> Never commit `.env`. It is git-ignored because environment files often hold secrets. Only `.env.example`, with placeholder or default values, belongs in the repository.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## API Usage Notes

- Without a configured key, `call_model` returns the supplied deterministic example. Provider failures remain visible and also return that example. Do not treat fallback responses as live results.
- Notebook 05 compares all three prompt conditions on six requests (18 answer calls plus one reviewer call for a complete live comparison). Simulated and fallback rows do not support a live architecture recommendation. The common rubric measures format, retrieval, citation-ID consistency, and concision, not sentence-level citation coverage, semantic faithfulness, or tone.
- A provider selected at startup is used for the run. Rate-limited calls and server errors are retried a few times with backoff before falling back; an API error does not switch to the other provider.
- Live calls may incur costs or hit rate limits. Check model availability and billing in the provider console, and restart the kernel after changing `.env`.
