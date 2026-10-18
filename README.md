---
title: WanderWise Travel Concierge
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# ✈️ WanderWise Travel — Aria AI Travel Concierge

**EECE 503P/798S | C3 – Introduction to Agents**  
**Assignment:** Build an Agent-Powered Business Assistant

WanderWise Travel is a **fictional** boutique travel agency (its packages and prices are invented for this classroom demonstration; the team listed in the knowledge files is the real student and course staff behind the project: Jad Elmoghraby, Prof. Ammar Mohanna, Zakaria El Labban and Mohamad Hamdan). Business contact: contact@jr-web-lb.com. Aria, its AI concierge, answers business questions, gathers qualified travel leads, and logs feedback or unanswered questions. This is **not** a live booking or travel-information service.

Companion repository for Assignment C4 (ReAct agent with configurable personas): [WanderWise-C4-ReAct-Agent](https://github.com/Jadjnm/WanderWise-C4-ReAct-Agent)

## Demo Video

**[Watch the C3 demo video](https://mailaub-my.sharepoint.com/:v:/g/personal/jne20_mail_aub_edu/IQBS758Xsy8gSI1HMyzuLB8-AYwz8GK7bNJh5nVka8Nwctc?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=T8ofLM)**

**[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/jadjnm/wanderwise-aria)**

The video walks through the business profile, the separate persona and instructions, the MCP-style tools, the intent router, and the live Gradio app: a general question, lead collection (saved with `record_customer_interest`), an unanswered question (logged with `record_feedback`), and the reflection pass.

## Multi-Provider Support

This project supports multiple LLM providers out of the box:
1. **Google Gemini**: via official `google-genai` SDK using `gemini-3.8-flash` (default).
2. **OpenRouter**: OpenAI-compatible endpoint with support for free models (default `dots-studio/dots-3-note-preview:free`, which passed 18/18 in the C4 model comparison) and budget models.
3. **OpenAI**: OpenAI Chat Completions with function calling (`gpt-4o-mini`).
4. **AWS Bedrock**: Boto3 Converse API with native tool use (`LLM_PROVIDER=bedrock`, plus `AWS_BEARER_TOKEN_BEDROCK`, IAM keys, or `AWS_PROFILE`; default `us.anthropic.claude-haiku-4-5-20251001-v1:0` / Claude Haiku 4.5). Also supported by the companion C4 ReAct agent.

The provider is automatically detected from the active keys in `.env`, or can be explicitly selected via `LLM_PROVIDER=gemini|openrouter|openai|bedrock`.

## Files and submission checklist

| File | Purpose |
|---|---|
| `business_agent.ipynb` | C3 notebook: persona, instructions, router, MCP tools, chat loop, and Gradio interface |
| `agent_core.py` | Multi-provider agent logic for C3 (router, MCP schemas, tools, LLM clients, reflection) |
| `app.py` | Gradio web application for C3 with live logs inspector |
| `check_connection.py` | Diagnostic utility to verify LLM provider API connectivity and credentials |
| `me/about_business.pdf` | Business profile (team, contact, services, fictional packages, policies) |
| `me/business_summary.txt` | Business knowledge used in system prompts (fictional agency; real project team) |
| `tests/test_agent.py` | Offline unit tests for C3 agent (30 tests; no live API key required) |
| `README.md` | Overview, architecture, setup instructions, and C3 documentation |
| `.env.example` | Template for environment variables and provider credentials |
| `requirements.txt` | Python package dependencies |

## Setup (Python 3.10+ recommended)

```bash
# 1. Clone repository and install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
```

Edit `.env` and provide your preferred key:
* **For Google Gemini:** Add `GEMINI_API_KEY=your_key` (via [Google AI Studio](https://aistudio.google.com/)).
* **For OpenRouter:** Add `OPENROUTER_API_KEY=your_key` (select free models or budget models).
* **For OpenAI:** Add `OPENAI_API_KEY=your_key`.
* **For AWS Bedrock:** Set `LLM_PROVIDER=bedrock` and provide Bedrock API token or IAM credentials.

Start the C3 application:

```bash
python app.py
```

Or open `business_agent.ipynb`, run its cells in order, and launch the interactive Gradio cell.

## Interaction and Routing Architecture

```text
Customer message + recent conversation history
               |
         Router (Rule & Pattern matching)
               |
   +-----------+---------------+
   |           |               |
general       lead     feedback_unanswered
   |           |               |
   +-----------+---------------+
               |
   Multi-Provider LLM (Gemini / OpenRouter / OpenAI / Bedrock)
        |                     |
  normal reply          model tool call
                        |             |
           record_customer_interest  record_feedback
                        |             |
                 local JSONL log files
                        |
                   model final reply
                        |
         optional reflection pass (ENABLE_REFLECTION=1)
                        |
       Gradio Web UI + Live Logs Inspector Tab
```

The router classifies incoming requests into **exactly three intents**:
1. `general_question`: Package details, pricing, team and contact information, and inclusions.
2. `lead_collection`: Booking interest, custom itinerary consultations, and contact information capture. When an email address is detected during lead collection, the agent forces the `record_customer_interest` tool call so that leads are recorded reliably.
3. `feedback_unanswered`: Customer feedback, complaints, or business-specific questions not answerable from approved materials (cancellation policies, refunds, flight booking, visa issuance, or unannounced future group tours). The router forces the `record_feedback` tool call so inquiries are logged without fabrication.

## Documented MCP-Style Function Tools

Tools are documented following the **Model Context Protocol (MCP)** specification:

1. `record_customer_interest(email: string, name: string, message: string)`
   - **Input Schema:** Validated email, customer name, and trip inquiry description.
   - **Output Schema:** `{"status": "success|error", "message": "..."}`.
   - **Log File:** Appends to `logs/leads.jsonl`.

2. `record_feedback(question: string)`
   - **Input Schema:** Exact customer question, complaint, or feedback.
   - **Output Schema:** `{"status": "success|error", "message": "..."}`.
   - **Log File:** Appends to `logs/feedback.jsonl`.

Full JSON schemas (`inputSchema`, `outputSchema`, required properties) are defined in `MCP_TOOL_DEFINITIONS` in `agent_core.py` and documented in the notebook.

## Reflection Pass

Set `ENABLE_REFLECTION=1` in `.env`.
Before the agent returns its final answer, a secondary critique pass verifies the response against the approved business documents and instructions. Original drafts and revised replies are saved to `logs/reflections.jsonl` so self-correction is fully visible.

## Testing

```bash
# Run C3 agent unit tests (30 tests)
python -m unittest tests/test_agent.py -v

# Or run every test in this repository
python -m unittest discover -s tests -v
```

The offline test suite runs 30 unit tests in `tests/test_agent.py` verifying:
* Routing across all 3 intents and unanswerable domain queries.
* Lead contact validation and local JSONL logging.
* Model function call execution and tool forcing.
* Formal MCP schema contracts (`inputSchema` and `outputSchema`).
* Provider auto-detection logic.

## Hugging Face Spaces Deployment

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) with the **Gradio** SDK.
2. Push the repository files to the Space.
3. Under **Space Settings > Variables and secrets**, add `GEMINI_API_KEY` (or `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / Bedrock credentials).
4. The Space will automatically build and launch `app.py`.
