"""WanderWise Pre-Flight Verification & Health Check.

Run this script before recording your demo video to verify API credentials,
local knowledge files, tool-calling permissions, and test suite health.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import agent_core as core


def print_status(component: str, ok: bool, details: str = ""):
    icon = "[OK]" if ok else "[WARN]"
    print(f"{icon} | {component:<30} | {details}")


def run_health_check():
    print("=" * 70)
    print(">> WANDERWISE PRE-FLIGHT SYSTEM HEALTH CHECK")
    print("=" * 70)

    # 1. Check local knowledge files
    summary_ok = (ROOT_DIR / "me" / "business_summary.txt").is_file()
    pdf_ok = (ROOT_DIR / "me" / "about_business.pdf").is_file()
    print_status("Business Summary (TXT)", summary_ok, "me/business_summary.txt")
    print_status("Business Profile (PDF)", pdf_ok, "me/about_business.pdf")

    try:
        knowledge = core.load_business_knowledge()
        print_status("Knowledge Extraction", len(knowledge) > 1000, f"{len(knowledge)} characters extracted")
    except Exception as err:
        print_status("Knowledge Extraction", False, str(err))

    # 2. Check logging directory permissions
    try:
        core.LOG_DIR.mkdir(parents=True, exist_ok=True)
        test_log = core.LOG_DIR / ".perm_check"
        test_log.write_text("ok", encoding="utf-8")
        test_log.unlink()
        print_status("Logs Directory (logs/)", True, "Writable JSONL directory verified")
    except Exception as err:
        print_status("Logs Directory (logs/)", False, str(err))

    # 3. Check active provider
    provider = core.get_active_provider()
    model = core.get_active_model(provider)
    print_status("Active LLM Provider", provider != "none", f"Provider: '{provider}', Model: '{model}'")

    # 4. Check API key status & optional live ping
    if provider == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        masked_key = gemini_key[:4] + "..." + gemini_key[-4:] if len(gemini_key) > 8 else "NOT SET"
        print_status("Google Gemini Key", bool(gemini_key), f"Key: {masked_key}")

        if gemini_key and core.genai:
            print("   -> Testing live connection to Google Gemini API (Free tier)...")
            try:
                client = core.genai.Client(api_key=gemini_key)
                resp = client.models.generate_content(
                    model=model,
                    contents="Say 'OK' in one word.",
                )
                print_status("Gemini Live API Ping", True, f"Response: {resp.text.strip()[:30]}")
            except Exception as e:
                print_status("Gemini Live API Ping", False, f"Error: {e}")

    elif provider == "openrouter":
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        masked_key = or_key[:4] + "..." + or_key[-4:] if len(or_key) > 8 else "NOT SET"
        print_status("OpenRouter Key", bool(or_key), f"Key: {masked_key}")

        if or_key:
            print("   -> Testing live connection to OpenRouter...")
            try:
                client = core.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=or_key,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'OK' in one word."}],
                    max_tokens=5,
                )
                print_status("OpenRouter Live Ping", True, f"Response: {resp.choices[0].message.content.strip()[:30]}")
            except Exception as e:
                print_status("OpenRouter Live Ping", False, f"Error: {e}")

    elif provider == "openai":
        oa_key = os.getenv("OPENAI_API_KEY", "")
        masked_key = oa_key[:4] + "..." + oa_key[-4:] if len(oa_key) > 8 else "NOT SET"
        print_status("OpenAI Key", bool(oa_key), f"Key: {masked_key}")

    else:
        print_status("API Key Configuration", False, "No key found in .env. Configure GEMINI_API_KEY or OPENROUTER_API_KEY.")

    # 5. MCP Schemas Verification
    mcp_tools = len(core.MCP_TOOL_DEFINITIONS)
    print_status("MCP Tool Contracts", mcp_tools == 2, f"{mcp_tools} tools with inputSchema & outputSchema")

    # 6. Run offline unit tests
    print("-" * 70)
    print("Running 13 offline unit tests...")
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT_DIR / "tests"))
    runner = unittest.TextTestRunner(verbosity=0)
    test_result = runner.run(suite)
    tests_ok = test_result.wasSuccessful()
    print_status("Unit Test Suite", tests_ok, f"{test_result.testsRun} tests run ({len(test_result.failures)} failures, {len(test_result.errors)} errors)")
    print("=" * 70)

    if tests_ok and provider != "none":
        print(">> System is verified and ready for live demo or video recording!")
        print("To launch the Gradio bot, run: python app.py")
    else:
        print(">> Note: To test live queries, add your GEMINI_API_KEY or OPENROUTER_API_KEY in .env.")
    print("=" * 70)


if __name__ == "__main__":
    run_health_check()
