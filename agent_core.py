"""WanderWise Travel: Aria, an educational agent-powered business assistant.

Course: EECE 503P/798S: Agentic Systems (Assignment C3)
Fictional business / demonstration data only. No real reservations are taken.
Supports: Google Gemini (Free tier), OpenRouter (Free/budget models), OpenAI, and AWS Bedrock.
"""

# ==============================================================================
# SECTION 1: Imports and Multi-Provider Configuration
# ==============================================================================
import json
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader

# Resilient base directory detection (works locally, in subdirectories, and in Google Colab)
def _find_base_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent if "__file__" in globals() else None,
        Path.cwd(),
        Path.cwd() / "WanderWise_Agent_Project",
        Path.cwd().parent,
    ]
    for c in candidates:
        if c and (c / "me" / "business_summary.txt").is_file():
            return c.resolve()
    return Path.cwd().resolve()

BASE_DIR = _find_base_dir()
load_dotenv(BASE_DIR / ".env")
LOG_DIR = BASE_DIR / "logs"

# Provider-specific SDK imports with graceful fallbacks
try:
    from openai import OpenAI
except ImportError:
    class OpenAI:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    import boto3
except ImportError:
    boto3 = None  # type: ignore


def get_active_provider() -> str:
    """Determine the active LLM provider based on .env configuration or present keys."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("gemini", "openrouter", "openai", "bedrock"):
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
        return "bedrock"
    return "none"


def get_active_model(provider: str | None = None) -> str:
    """Return the model name corresponding to the selected provider."""
    p = provider or get_active_provider()
    if p == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    if p == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "dots-studio/dots-3-note-preview:free")
    if p == "bedrock":
        return os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Backward-compatible global MODEL reference
MODEL = get_active_model()

# ==============================================================================
# SECTION 2: PERSONA (Identity, Tone, and Voice)
# ==============================================================================
PERSONA = """
You are Aria, the friendly, enthusiastic, and thoughtful virtual travel concierge
for WanderWise Travel, a fictional boutique travel-planning business.
You sound like a knowledgeable, well-traveled friend: welcoming, curious,
practical, and never pushy. Your writing is warm and concise. A sparing travel
emoji is fine, but clarity matters more than cheerleading.
""".strip()

# ==============================================================================
# SECTION 3: INSTRUCTIONS (Operational Guardrails)
# ==============================================================================
INSTRUCTIONS = """
1. Stay in character as Aria from WanderWise Travel.
2. Use ONLY the supplied business summary and business PDF extracts for factual
   claims about WanderWise: team, prices, packages, services, inclusions,
   policies, and contact methods. Never make up deals, dates, pricing, visas,
   availability, or reservation status.
3. For general inquiries, answer clearly using approved facts. When appropriate,
   ask about budget, interests, travel dates, and number of travelers, one or two at a time.
4. When someone wants a quote, consultation, or to book a trip, collect their name
   and email voluntarily, along with a brief description of their trip. Tell them this
   is an inquiry consultation, not a confirmed live booking. Once required details
   are given, call record_customer_interest(email, name, message). NEVER invent an email
   or name; politely ask if any required field is missing.
5. If the user submits a complaint, feedback, or asks an UNANSWERED business-specific
   question (e.g. cancellation policy, refunds, flight booking, visa issuance, insurance,
   or unannounced future tour departures), call record_feedback(question) with the exact
   question or feedback. Explain the boundary honestly instead of guessing.
6. Never request or store passwords, credit cards, passport numbers, or bank details.
7. You have no live flight, hotel inventory, payment, or visa-issuing tools. Never claim
   you booked a flight or confirmed availability.
8. If a logging tool returns an error, acknowledge it; NEVER say the inquiry or feedback
   was recorded unless the tool reports success.
9. This is an academic class demonstration using invented offerings, not a real agency.
""".strip()

# ==============================================================================
# SECTION 4: Business Knowledge (TXT + PDF Loading)
# ==============================================================================
@lru_cache(maxsize=1)
def load_business_knowledge() -> str:
    """Read both assignment-required knowledge files without external retrieval."""
    summary_path = BASE_DIR / "me" / "business_summary.txt"
    pdf_path = BASE_DIR / "me" / "about_business.pdf"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing business summary: {summary_path}")

    summary = summary_path.read_text(encoding="utf-8")
    pdf_content = ""
    if pdf_path.is_file():
        try:
            pdf_content = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
        except Exception as err:
            pdf_content = f"[Note: PDF extract error: {err}]"

    return (
        "BUSINESS SUMMARY (primary reference):\n" + summary +
        "\n\nBUSINESS PDF EXTRACT (supporting reference):\n" + pdf_content[:16000]
    )


def build_system_prompt(intent: str) -> str:
    """Assemble Persona, Instructions, Router Directive, and Approved Knowledge."""
    directives = {
        "general_question": "Route: general question. Answer from the approved knowledge, or call record_feedback if information is missing.",
        "lead_collection": "Route: lead collection. Ask for missing name/email/notes; call record_customer_interest when details are complete.",
        "feedback_unanswered": "Route: feedback or unanswered business question. Call record_feedback to record the query before giving a polite answer.",
    }
    note = directives.get(intent, directives["general_question"])
    return (
        "PERSONA\n" + PERSONA +
        "\n\nINSTRUCTIONS\n" + INSTRUCTIONS +
        "\n\nROUTER DECISION\n" + note +
        "\n\nAPPROVED BUSINESS KNOWLEDGE\n" + load_business_knowledge()
    )

# ==============================================================================
# SECTION 5: Formal MCP-Style Tool Schemas
# ==============================================================================
# Documented in Model Context Protocol (MCP) format: name, description, inputSchema, outputSchema.
MCP_TOOL_DEFINITIONS = [
    {
        "name": "record_customer_interest",
        "description": "Log a customer's voluntarily supplied contact details and vacation inquiry notes. Returns a structured JSON result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "Customer's valid email address for follow-up.",
                },
                "name": {
                    "type": "string",
                    "description": "Full name of the inquiring customer.",
                },
                "message": {
                    "type": "string",
                    "description": "Summary of the trip details, destination, dates, budget, or travel style.",
                },
            },
            "required": ["email", "name", "message"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"], "description": "Outcome of the operation"},
                "message": {"type": "string", "description": "Human-readable confirmation message"},
            },
            "required": ["status", "message"],
        },
    },
    {
        "name": "record_feedback",
        "description": "Log customer feedback, complaints, suggestions, or unanswered business-specific questions not found in WanderWise documentation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The exact customer question, suggestion, or complaint to record.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"], "description": "Outcome of the operation"},
                "message": {"type": "string", "description": "Human-readable confirmation message"},
            },
            "required": ["status", "message"],
        },
    },
]

# Standard OpenAI / OpenRouter function calling format derived directly from MCP definitions
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        },
    }
    for tool in MCP_TOOL_DEFINITIONS
]

# ==============================================================================
# SECTION 6: Real Logging Functions / Tool Implementations
# ==============================================================================
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def _append_log(filename: str, data: dict[str, Any]) -> None:
    """Append structured record to a local JSONL file; folder is excluded from Git."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / filename).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def record_customer_interest(email: str, name: str, message: str) -> dict[str, str]:
    """Log lead; inputs: email, name, message; output: {status, message}."""
    email, name, message = email.strip(), name.strip(), message.strip()
    # Require real content, not just non-empty strings (models have sent ", " or ": " as values)
    if (not EMAIL_RE.fullmatch(email) or len(re.findall(r"[A-Za-z]", name)) < 2
            or len(re.findall(r"[A-Za-z]", message)) < 3):
        return {"status": "error", "message": "A valid email, name, and trip description are required."}
    _append_log("leads.jsonl", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "message": message,
    })
    print(f"[TOOL] record_customer_interest: lead for '{name}' saved to logs/leads.jsonl")
    return {"status": "success", "message": "Travel inquiry saved for the demo team; no trip is booked."}


def record_feedback(question: str) -> dict[str, str]:
    """Log unknown business question or feedback; input: question; output: {status, message}."""
    question = question.strip()
    if not question:
        return {"status": "error", "message": "Question or feedback must not be empty."}
    _append_log("feedback.jsonl", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
    })
    print(f"[TOOL] record_feedback: entry saved to logs/feedback.jsonl")
    return {"status": "success", "message": "Question/feedback recorded; no answer was invented."}


TOOL_FUNCTIONS = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}

# ==============================================================================
# SECTION 7: Message-Intent Router
# ==============================================================================
def _plain_text(content: Any) -> str:
    """Extract plain string from Gradio history format or plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(block.get("text", "")) for block in content
                        if isinstance(block, dict) and block.get("type") == "text")
    return ""


def route_message_with_metadata(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Classify message into exactly 3 intents with explanation and matched rule."""
    lower = message.lower().strip()
    recent = history[-6:] if history else []
    recent_text = " ".join(_plain_text(item.get("content", ""))
                           for item in recent if item.get("role") in {"user", "assistant"}).lower()

    # Rule 1: Explicit feedback, complaint, or customer service dissatisfaction
    if re.search(r"\b(complaint|feedback|suggestion|unhappy|disappointed|report a problem|poor service|bug report)\b", lower):
        return {
            "intent": "feedback_unanswered",
            "reason": "Customer expressed feedback, suggestion, or complaint.",
            "rule": "explicit_feedback",
        }

    # Rule 2: Unanswerable business questions explicitly documented in FAQ and policies
    # Cancellation, refund, live visa issuance, flight booking, insurance, card charging
    if re.search(r"\b(cancellation|cancellations|cancel|cancellation policy|refund|refunds|refund policy)\b", lower):
        return {
            "intent": "feedback_unanswered",
            "reason": "Cancellation/refund policies are not defined in business materials.",
            "rule": "unsupported_cancellation_policy",
        }
    if re.search(r"\b(visa|visas|entry permit|passport renewal)\b", lower) and re.search(r"\b(issue|apply|give me|procure|provide|guarantee)\b", lower):
        return {
            "intent": "feedback_unanswered",
            "reason": "Visa issuance and government immigration cannot be provided.",
            "rule": "unsupported_visa_issuance",
        }
    if re.search(r"\b(travel insurance|medical insurance)\b", lower):
        return {
            "intent": "feedback_unanswered",
            "reason": "Travel insurance is excluded from all packages.",
            "rule": "unsupported_insurance",
        }

    # Rule 3: Questions about future, unannounced offerings (Japan group tours, next year launches)
    if (re.search(r"\b(will you|will wanderwise|are you|is wanderwise)\b", lower)
            and re.search(r"\b(launch|offer|open|opening|group tours?|group departures?)\b", lower)):
        return {
            "intent": "feedback_unanswered",
            "reason": "Inquiry about unannounced future tour offerings.",
            "rule": "future_unannounced_offering",
        }
    if "next year" in lower and re.search(r"\b(group tours?|group departures?|launch|announc|open)\b", lower):
        return {
            "intent": "feedback_unanswered",
            "reason": "Inquiry about unannounced departures next year.",
            "rule": "future_unannounced_offering",
        }

    # Rule 4: Name/email follow-up during an established lead consultation
    if EMAIL_RE.search(message) and any(w in recent_text for w in ["name and email", "book", "quote", "contact", "consultation", "details"]):
        return {
            "intent": "lead_collection",
            "reason": "Customer supplied contact details in response to consultation follow-up.",
            "rule": "lead_contact_followup",
        }

    # Rule 5: Expressions of booking interest, quotes, or custom planning
    if re.search(r"\b(book|booking|reserve|reservation|quote|consultation|sign me up|contact me|interested in booking|ready to book|plan a trip|plan my trip|custom itinerary)\b", lower):
        return {
            "intent": "lead_collection",
            "reason": "Customer expressed interest in booking or scheduling a consultation.",
            "rule": "lead_intent_detected",
        }

    # Default Rule: General questions about packages, team, or services
    return {
        "intent": "general_question",
        "reason": "General question regarding packages, pricing, team, or services.",
        "rule": "general_knowledge_query",
    }


def route_message(message: str, history: list[dict[str, Any]] | None = None) -> str:
    """Return exactly one of three intent labels for backward compatibility."""
    return route_message_with_metadata(message, history)["intent"]

# ==============================================================================
# SECTION 8: Multi-Provider Model Interaction & Tool Execution
# ==============================================================================
def _chat_messages(message: str, history: list[dict[str, Any]] | None, intent: str) -> list[dict[str, Any]]:
    """Build standardized OpenAI-format messages list."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(intent)}]
    for item in (history or [])[-12:]:
        role = item.get("role")
        if role in ("user", "assistant"):
            body = _plain_text(item.get("content", ""))
            if body:
                messages.append({"role": role, "content": body})
    messages.append({"role": "user", "content": message})
    return messages


_NAME_PATTERNS = [
    re.compile(r"\b(?:i'?m|i am|my name is|this is|name\s*[:=-])\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})", re.I),
    re.compile(r"\b([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){1,3})\s*[,(-]\s*[\w.%+-]+@"),  # "Dana Saleh, dana@..."
]


def _user_turns(message: str, history: list[dict[str, Any]] | None) -> list[str]:
    turns = [_plain_text(h.get("content", "")) for h in (history or [])[-8:] if h.get("role") == "user"]
    return [t for t in turns + [message] if t]


def _repair_lead_args(args: dict[str, Any], message: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Some models send empty or junk arguments for a forced record_customer_interest call even when the
    customer gave the details. Fill only the invalid fields, and only from the customer's own words:
    the email and name are copied verbatim, and the trip note quotes their messages. A field that isn't
    in the conversation stays invalid, so the tool still refuses and the model asks for it."""
    args = dict(args)
    turns = _user_turns(message, history)
    text = "\n".join(turns)
    if not EMAIL_RE.fullmatch(str(args.get("email", "")).strip()):
        found = EMAIL_RE.findall(text)
        args["email"] = found[-1] if found else ""
    if not _has_letters(args.get("name"), 2):
        args["name"] = ""
        for turn in reversed(turns):
            match = next((m for p in _NAME_PATTERNS if (m := p.search(turn))), None)
            if match:
                args["name"] = match.group(1).strip()
                break
    if not _has_letters(args.get("message"), 3):
        notes = []
        for turn in turns:
            note = EMAIL_RE.sub("", turn)
            for pattern in _NAME_PATTERNS:
                note = pattern.sub("", note)
            note = re.sub(r"\b(my )?(e-?mail|name)( address)?\s*(is|:)?", "", note, flags=re.I).strip(" ,.;:-|")
            if _has_letters(note, 10):
                notes.append(note)
        args["message"] = " | ".join(notes)[-400:]
    return args


def _has_letters(value: Any, minimum: int) -> bool:
    return len(re.findall(r"[A-Za-z]", str(value or ""))) >= minimum


def _run_tool(name: str, arguments: Any, message: str = "", history: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """Execute a model-requested tool safely; lead arguments are repaired from the conversation first."""
    if name not in TOOL_FUNCTIONS:
        return {"status": "error", "message": f"Unknown tool: {name}"}
    try:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        if name == "record_customer_interest":
            repaired = _repair_lead_args(arguments, message, history)
            if repaired != arguments:
                print("[TOOL] record_customer_interest: model arguments were incomplete; filled from the customer's messages")
            arguments = repaired
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as exc:
        return {"status": "error", "message": f"Logging tool execution failed: {exc}"}


def _execute_tool_call(call: Any, message: str = "", history: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """Execute an OpenAI-format tool call safely and return structured result."""
    try:
        arguments = json.loads(call.function.arguments)
    except Exception as exc:
        return {"status": "error", "message": f"Logging tool execution failed: {exc}"}
    return _run_tool(getattr(call.function, "name", ""), arguments, message, history)


def _log_tool_result(name: str, result: dict[str, str]) -> None:
    detail = "" if result.get("status") == "success" else f" ({result.get('message', '')})"
    print(f"[TOOL RESULT] {name}: {result.get('status')}{detail}")


def _openai_client(provider: str) -> Any:
    """OpenAI-SDK client for OpenRouter or OpenAI, or None when its API key is missing."""
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={"HTTP-Referer": "https://wanderwise.local", "X-Title": "WanderWise Aria Concierge"},
        )
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


def reflect_on_answer(draft: str, intent: str, client: Any = None, provider: str = "openai") -> str:
    """Optional bonus: critique draft against instructions and knowledge; log both."""
    critique_prompt = (
        "REFLECTION CHECK: Review the draft assistant reply below. Rewrite it ONLY if needed "
        "to remove unsupported WanderWise claims, invented prices/availability, or a missed "
        "request for contact information during a lead consultation. Preserve actual tool logging "
        "status exactly. Tools are not available in this step: do not call or write out any tool or "
        "function call. Return ONLY the final revised customer-facing reply as plain text."
    )
    revised = draft
    try:
        if provider == "gemini" and genai:
            gem_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = gem_client.models.generate_content(
                model=get_active_model("gemini"),
                contents=[
                    genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=f"Draft answer:\n{draft}\n\n{critique_prompt}")]),
                ],
            )
            revised = (response.text or draft).strip()
        elif provider == "bedrock" and client is None:
            response = _bedrock_client().converse(
                modelId=get_active_model("bedrock"),
                system=[{"text": build_system_prompt(intent)}],
                messages=[{"role": "user", "content": [{"text": f"Draft answer:\n{draft}\n\n{critique_prompt}"}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 800},
            )
            blocks = response["output"]["message"].get("content", [])
            revised = ("".join(b.get("text", "") for b in blocks) or draft).strip()
        elif client is not None or provider in ("openrouter", "openai"):
            # Build the same client the chat path uses when the caller didn't pass one (notebook, app)
            client = client or _openai_client(provider)
            if client is None:
                raise RuntimeError(f"no API key configured for {provider}")
            reflection = _create_completion(
                client,
                model=get_active_model(provider),
                messages=[
                    {"role": "system", "content": build_system_prompt(intent)},
                    {"role": "user", "content": f"Draft answer:\n{draft}\n\n{critique_prompt}"},
                ],
            )
            revised = (reflection.choices[0].message.content or draft).strip()
    except Exception as err:
        print(f"[BONUS] Reflection check skipped due to error: {err}")
        revised = draft
    # A model that answers with tool-call markup instead of a reply gives us nothing to show the customer
    if re.search(r"<\s*/?\s*(\w+_)?(invoke|function_call|tool_call)\b", revised, flags=re.I):
        print("[BONUS] Reflection returned tool-call markup instead of a reply; keeping the original draft.")
        revised = draft
    # Models sometimes prefix the rewrite with a label the customer should never see
    revised = re.sub(r"^\W*(revised|final|corrected)\s+(reply|answer|response)\W*\s*", "", revised, flags=re.I).strip() or draft

    _append_log("reflections.jsonl", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_draft": draft,
        "revised_reply": revised,
    })
    print("[BONUS] Reflection logged (original draft and revised reply saved to logs/reflections.jsonl).")
    return revised


def _run_gemini(message: str, history: list[dict[str, Any]] | None, intent: str) -> tuple[str, list[dict[str, Any]]]:
    """Execute query using the native Google GenAI SDK (Free Gemini API)."""
    if not genai:
        return ("Google GenAI SDK is not installed. Please install with: pip install google-genai", [])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ("Please add your free GEMINI_API_KEY in the .env file to chat with Aria via Google Gemini.", [])

    model_name = get_active_model("gemini")
    client = genai.Client(api_key=api_key)
    system_prompt = build_system_prompt(intent)
    tools_called = []

    def wrapped_record_customer_interest(email: str, name: str, message: str) -> dict[str, str]:
        res = record_customer_interest(email, name, message)
        tools_called.append({"name": "record_customer_interest", "args": {"email": email, "name": name, "message": message}, "result": res})
        return res

    def wrapped_record_feedback(question: str) -> dict[str, str]:
        res = record_feedback(question)
        tools_called.append({"name": "record_feedback", "args": {"question": question}, "result": res})
        return res

    # Assemble conversation contents
    contents: list[genai_types.Content] = []
    for item in (history or [])[-8:]:
        role = item.get("role")
        body = _plain_text(item.get("content", ""))
        if body and role in ("user", "assistant"):
            c_role = "user" if role == "user" else "model"
            contents.append(genai_types.Content(role=c_role, parts=[genai_types.Part.from_text(text=body)]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)]))

    # If routed to feedback_unanswered, force feedback tool for Gemini
    if intent == "feedback_unanswered" and genai_types:
        tool_config = genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode=genai_types.FunctionCallingConfigMode.ANY,
                allowed_function_names=["record_feedback"],
            )
        )
    else:
        tool_config = None

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[wrapped_record_customer_interest, wrapped_record_feedback],
        tool_config=tool_config,
        temperature=0.2,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        reply = (response.text or "").strip()
        if not reply:
            reply = "I've recorded your details and our travel team will review your inquiry shortly!"
        return reply, tools_called
    except Exception as exc:
        print(f"[GEMINI ERROR] {type(exc).__name__}: {exc}")
        return (f"Encountered an issue communicating with Google Gemini: {exc}", tools_called)


def _create_completion(client: Any, attempts: int = 3, **kwargs: Any) -> Any:
    """chat.completions.create with retries for transient provider failures: rate limits, 5xx,
    and OpenRouter's HTTP-200 responses that carry an error body instead of choices."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            if getattr(response, "choices", None):
                return response
            problem = f"provider returned no choices: {getattr(response, 'error', None) or 'empty response'}"
        except Exception as exc:
            text = str(exc)
            if not any(code in text for code in ("429", "500", "502", "503", "504", "rate", "timeout", "Timeout")):
                raise
            problem = text[:200]
        if attempt < attempts:
            print(f"[RETRY] {problem} (attempt {attempt}/{attempts})")
            time.sleep(3 * attempt)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {problem}")


def _run_openai_compatible(message: str, history: list[dict[str, Any]] | None, intent: str,
                           client: Any = None, provider: str = "openai") -> tuple[str, list[dict[str, Any]]]:
    """Execute query using OpenAI SDK (compatible with OpenRouter, OpenAI, and Local Mocks)."""
    model_name = get_active_model(provider)
    tools_called = []

    if client is None:
        client = _openai_client(provider)
        if client is None:
            if provider == "openrouter":
                return ("Please add your OPENROUTER_API_KEY in the .env file to use OpenRouter.", [])
            return ("Please add your OPENAI_API_KEY in the .env file to use OpenAI.", [])

    messages = _chat_messages(message, history, intent)
    # The router decides when a tool is mandatory: feedback/unanswered questions always get logged,
    # and a lead message that carries an email always gets saved. Leaving these to "auto" let models
    # reply "I've logged your inquiry" without calling the tool.
    forced = _forced_tool(message, intent)
    tool_choice: Any = {"type": "function", "function": {"name": forced}} if forced else "auto"

    response = _create_completion(
        client,
        model=model_name,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice=tool_choice,
    )
    assistant = response.choices[0].message
    calls = assistant.tool_calls or []
    if forced and not calls:
        # Some models ignore a forced tool_choice. The router already decided this call is required,
        # so run it here (lead arguments are then filled only from the customer's own messages).
        print(f"[TOOL] {forced}: the model skipped the required call; running it from the router")
        args = {"question": message} if forced == "record_feedback" else {}
        calls = [SimpleNamespace(id="router_required_call",
                                 function=SimpleNamespace(name=forced, arguments=json.dumps(args)))]

    if calls:
        messages.append({
            "role": "assistant",
            "content": assistant.content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in calls
            ],
        })
        for call in calls:
            result = _execute_tool_call(call, message, history)
            _log_tool_result(call.function.name, result)
            tools_called.append({
                "name": call.function.name,
                "arguments": getattr(call.function, "arguments", "{}"),
                "result": result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
        final_response = _create_completion(client, model=model_name, messages=messages)
        draft = (final_response.choices[0].message.content or "").strip()
    else:
        draft = (assistant.content or "").strip()

    if not draft:
        draft = "Could you tell me a little more about your travel plans so I can help?"
    return draft, tools_called


def _forced_tool(message: str, intent: str) -> str | None:
    """Tool the router makes mandatory for this message (shared by all providers)."""
    if intent == "feedback_unanswered":
        return "record_feedback"
    if intent == "lead_collection" and EMAIL_RE.search(message):
        return "record_customer_interest"
    return None


_BEDROCK_CLIENT: Any = None


def _bedrock_client() -> Any:
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed. Run: pip install boto3")
        from botocore.config import Config
        # boto3 reads AWS_BEARER_TOKEN_BEDROCK or the access-key pair from the environment
        _BEDROCK_CLIENT = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION") or "us-east-1",
            config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=120),
        )
    return _BEDROCK_CLIENT


def _bedrock_messages(message: str, history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Converse needs strictly alternating user/assistant turns that start with the user."""
    turns: list[dict[str, Any]] = []
    for item in [*(history or [])[-12:], {"role": "user", "content": message}]:
        role, body = item.get("role"), _plain_text(item.get("content", ""))
        if role not in ("user", "assistant") or not body:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"][0]["text"] += "\n\n" + body
        else:
            turns.append({"role": role, "content": [{"text": body}]})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


BEDROCK_TOOL_CONFIG_TOOLS = [
    {"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": t["inputSchema"]}}}
    for t in MCP_TOOL_DEFINITIONS
]


def _run_bedrock(message: str, history: list[dict[str, Any]] | None, intent: str) -> tuple[str, list[dict[str, Any]]]:
    """Execute query with AWS Bedrock's Converse API and native tool use."""
    client = _bedrock_client()
    model_id = get_active_model("bedrock")
    system = [{"text": build_system_prompt(intent)}]
    messages = _bedrock_messages(message, history)
    forced = _forced_tool(message, intent)
    tool_choice = {"tool": {"name": forced}} if forced else {"auto": {}}
    tools_called: list[dict[str, Any]] = []

    response = client.converse(
        modelId=model_id, system=system, messages=messages,
        toolConfig={"tools": BEDROCK_TOOL_CONFIG_TOOLS, "toolChoice": tool_choice},
        inferenceConfig={"temperature": 0.2, "maxTokens": 800},
    )
    assistant = response["output"]["message"]
    tool_uses = [b["toolUse"] for b in assistant.get("content", []) if "toolUse" in b]

    if tool_uses:
        results = []
        for use in tool_uses:
            name, args = use["name"], use.get("input") or {}
            result = _run_tool(name, args, message, history)
            _log_tool_result(name, result)
            tools_called.append({"name": name, "arguments": json.dumps(args), "result": result})
            results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}]}})
        messages += [assistant, {"role": "user", "content": results}]
        # Second turn: let the model write the reply now that the tool has run (no forcing, or it loops)
        response = client.converse(
            modelId=model_id, system=system, messages=messages,
            toolConfig={"tools": BEDROCK_TOOL_CONFIG_TOOLS, "toolChoice": {"auto": {}}},
            inferenceConfig={"temperature": 0.2, "maxTokens": 800},
        )
        assistant = response["output"]["message"]

    draft = "".join(b.get("text", "") for b in assistant.get("content", [])).strip()
    if not draft:
        draft = "Could you tell me a little more about your travel plans so I can help?"
    return draft, tools_called


def run_agent(message: str, history: list[dict[str, Any]] | None = None,
              client: Any = None, provider: str | None = None,
              enable_reflection: bool | None = None) -> str:
    """Unified entry point: Route -> Multi-Provider LLM -> Execute Tools -> Optional Reflection."""
    meta = run_agent_with_metadata(
        message=message,
        history=history,
        client=client,
        provider=provider,
        enable_reflection=enable_reflection,
    )
    return meta["reply"]


def run_agent_with_metadata(message: str, history: list[dict[str, Any]] | None = None,
                            client: Any = None, provider: str | None = None,
                            enable_reflection: bool | None = None) -> dict[str, Any]:
    """Run agent and return response together with execution telemetry for Gradio UI."""
    if not message or not message.strip():
        return {
            "reply": "Tell me where you'd like to travel, and I'll help you explore the options!",
            "intent": "general_question",
            "provider": "none",
            "model": "none",
            "tools_called": [],
            "reflected": False,
        }

    active_provider = provider or ("openai" if client is not None else get_active_provider())
    enable_ref = (os.getenv("ENABLE_REFLECTION", "0") == "1") if enable_reflection is None else enable_reflection

    route_info = route_message_with_metadata(message, history)
    intent = route_info["intent"]
    print(f"[ROUTER] Intent: {intent} (Rule: {route_info['rule']})")

    # Dispatch to appropriate provider
    if active_provider == "gemini" and client is None:
        draft, tools_called = _run_gemini(message, history, intent)
    elif active_provider in ("openrouter", "openai") or client is not None:
        draft, tools_called = _run_openai_compatible(message, history, intent, client=client, provider=active_provider)
    elif active_provider == "bedrock":
        draft, tools_called = _run_bedrock(message, history, intent)
    elif active_provider == "none":
        return {
            "reply": (
                "👋 Welcome to WanderWise! To chat with Aria, please configure your API key in the `.env` file.\n\n"
                "• **Google Gemini (Free):** Set `GEMINI_API_KEY` (Free at https://aistudio.google.com/)\n"
                "• **OpenRouter:** Set `OPENROUTER_API_KEY` (Supports free models)\n"
                "• **OpenAI:** Set `OPENAI_API_KEY`"
            ),
            "intent": intent,
            "provider": "none",
            "model": "none",
            "tools_called": [],
            "reflected": False,
        }
    else:
        draft, tools_called = _run_openai_compatible(message, history, intent, client=client, provider="openai")

    # Reflection pass (Bonus)
    final_reply = draft
    if enable_ref and not draft.startswith("Please add your") and not draft.startswith("Encountered an issue"):
        final_reply = reflect_on_answer(draft=draft, intent=intent, client=client, provider=active_provider)

    return {
        "reply": final_reply,
        "intent": intent,
        "reason": route_info["reason"],
        "provider": active_provider,
        "model": get_active_model(active_provider),
        "tools_called": tools_called,
        "reflected": enable_ref,
    }

# ==============================================================================
# SECTION 9: Gradio Web UI Integration
# ==============================================================================
def chat(message: str, history: list[dict[str, Any]], enable_reflection: bool = False) -> str:
    """Standard Gradio ChatInterface callback with optional live reflection pass."""
    try:
        return run_agent(message, history, enable_reflection=enable_reflection)
    except Exception as exc:
        print(f"[AGENT ERROR] {type(exc).__name__}: {exc}")
        return ("I'm having trouble connecting right now. Please verify your API key, "
                "internet connection, and selected model in `.env`.")


def build_demo():
    """Build enhanced Gradio application with Live Log Inspector and Telemetry."""
    import gradio as gr

    def load_jsonl_records(filename: str) -> list[dict[str, Any]]:
        path = LOG_DIR / filename
        if not path.is_file():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records[::-1]  # Most recent first

    def format_logs_markdown(leads: list[dict], feedback: list[dict], reflections: list[dict]) -> str:
        md = []
        md.append("### 📥 Recorded Leads (`logs/leads.jsonl`)\n")
        if not leads:
            md.append("*No leads recorded yet.*")
        else:
            for l in leads:
                md.append(f"- **{l.get('name', 'Unknown')}** (`{l.get('email', '')}`): {l.get('message', '')} *(at {l.get('timestamp_utc', '')[:19]})*")

        md.append("\n---\n### 📝 Recorded Feedback & Unanswered Questions (`logs/feedback.jsonl`)\n")
        if not feedback:
            md.append("*No feedback recorded yet.*")
        else:
            for f in feedback:
                md.append(f"- ❓ \"{f.get('question', '')}\" *(at {f.get('timestamp_utc', '')[:19]})*")

        md.append("\n---\n### 🔄 Reflection Passes (`logs/reflections.jsonl`)\n")
        if not reflections:
            md.append("*No reflections recorded yet (check 'Enable Reflection Pass' or set `ENABLE_REFLECTION=1` in `.env`).*")
        else:
            for r in reflections[:3]:
                md.append(f"**Original Draft:**\n> {r.get('original_draft', '')}\n\n**Revised Reply:**\n> {r.get('revised_reply', '')}\n")
        return "\n".join(md)

    with gr.Blocks(title="✈️ Aria | WanderWise Travel Concierge") as demo:
        gr.Markdown(
            "# ✈️ WanderWise Travel — Aria AI Travel Concierge\n"
            "**EECE 503P/798S | Assignment C3: Build an Agent-Powered Business Assistant**\n\n"
            f"*Active Provider:* **{get_active_provider().upper()}** (`{get_active_model()}`) | "
            f"*Classroom simulation: All packages and prices are fictional.*"
        )

        with gr.Tabs():
            with gr.TabItem("💬 Travel Concierge Chat"):
                chat_interface = gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[
                        gr.Checkbox(label="Enable Reflection Pass (Self-Correction Check)", value=False),
                    ],
                    examples=[
                        ["Hi Aria! What packages do you offer and what is included in Bali Bliss?", False],
                        ["I'd like to book a consultation for a honeymoon to Bali in June.", False],
                        ["What is your cancellation and refund policy?", False],
                        ["Will WanderWise launch Japan group tours next year?", False],
                        ["I have feedback: please add more budget-friendly trips.", False],
                    ],
                )

            with gr.TabItem("📊 Live Logs & Leads Inspector"):
                gr.Markdown("Inspect in real time what Aria has logged via tool calls during your demonstration.")
                refresh_btn = gr.Button("🔄 Refresh Live Logs", variant="primary")
                logs_display = gr.Markdown(
                    value=lambda: format_logs_markdown(
                        load_jsonl_records("leads.jsonl"),
                        load_jsonl_records("feedback.jsonl"),
                        load_jsonl_records("reflections.jsonl"),
                    )
                )
                refresh_btn.click(
                    fn=lambda: format_logs_markdown(
                        load_jsonl_records("leads.jsonl"),
                        load_jsonl_records("feedback.jsonl"),
                        load_jsonl_records("reflections.jsonl"),
                    ),
                    outputs=logs_display,
                )

            with gr.TabItem("📋 Business Identity & Guardrails"):
                with gr.Accordion("🏢 Approved Business Summary (me/business_summary.txt)", open=True):
                    gr.Markdown(f"```text\n{load_business_knowledge()[:3000]}\n...\n```")
                with gr.Accordion("🛡️ Persona & Operational Instructions", open=False):
                    gr.Markdown(f"**PERSONA:**\n```text\n{PERSONA}\n```\n\n**INSTRUCTIONS:**\n```text\n{INSTRUCTIONS}\n```")
                with gr.Accordion("⚙️ MCP Tool Definitions", open=False):
                    gr.JSON(value=MCP_TOOL_DEFINITIONS)

    return demo


if __name__ == "__main__":
    build_demo().launch()
