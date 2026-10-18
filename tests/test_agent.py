"""Offline tests. Do not require an OpenAI API key or network access."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_core as core


def mock_message(content='', calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def mock_response(content='', calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=mock_message(content, calls))])


class FakeCompletions:
    def __init__(self, scripted):
        self.scripted = iter(scripted)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.scripted)


class AgentTests(unittest.TestCase):
    def test_router_general(self):
        self.assertEqual(core.route_message('What is included in Bali Bliss?'), 'general_question')

    def test_router_lead(self):
        self.assertEqual(core.route_message('I want to book a consultation'), 'lead_collection')

    def test_router_feedback(self):
        self.assertEqual(core.route_message('Will you launch group tours next year?'), 'feedback_unanswered')

    def test_router_company_named_future_question(self):
        self.assertEqual(core.route_message('Will WanderWise launch Japan group tours next year?'),
                         'feedback_unanswered')

    def test_router_followup_email(self):
        history = [
            {'role': 'user', 'content': 'I want to book a honeymoon'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'May I have your name and email?'}]},
        ]
        self.assertEqual(core.route_message('Dana Saleh, dana@example.com', history), 'lead_collection')

    def test_summary_and_pdf_both_loaded(self):
        content = core.load_business_knowledge()
        self.assertIn('BUSINESS SUMMARY', content)
        self.assertIn('BUSINESS PDF EXTRACT', content)
        self.assertIn('Bali Bliss', content)
        self.assertIn('Jad Elmoghraby', content)
        # The PDF extract must also carry the new team (proves the PDF was regenerated).
        pdf_part = content.split('BUSINESS PDF EXTRACT', 1)[1]
        self.assertIn('Jad Elmoghraby', pdf_part)

    def test_no_old_team_names_in_knowledge(self):
        content = core.load_business_knowledge()
        for old in ('Maya Haddad', 'Omar Nasser', 'Lina Farah'):
            self.assertNotIn(old, content)

    def test_lead_requires_valid_fields(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            result = core.record_customer_interest('oops', 'Dana', 'Bali trip')
            self.assertEqual(result['status'], 'error')
            self.assertFalse((Path(tmp)/'leads.jsonl').exists())

    def test_tools_log_to_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            one = core.record_customer_interest('dana@example.com', 'Dana', 'Bali honeymoon')
            two = core.record_feedback('What is your cancellation policy?')
            self.assertEqual(one['status'], 'success')
            self.assertEqual(two['status'], 'success')
            lead = json.loads((Path(tmp)/'leads.jsonl').read_text().splitlines()[0])
            feedback = json.loads((Path(tmp)/'feedback.jsonl').read_text().splitlines()[0])
            self.assertEqual(lead['email'], 'dana@example.com')
            self.assertEqual(feedback['question'], 'What is your cancellation policy?')

    def test_model_tool_call_executes_and_returns_result(self):
        call = SimpleNamespace(
            id='fake-call-1',
            function=SimpleNamespace(name='record_customer_interest',
                arguments=json.dumps({'email':'dana@example.com','name':'Dana',
                                      'message':'Bali honeymoon for two'})),
        )
        mock = FakeCompletions([
            mock_response(calls=[call]),
            mock_response('Thanks, Dana! Your inquiry was saved, but nothing is booked.'),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            output = core.run_agent('I want to book a Bali trip. I am Dana, dana@example.com',
                                    client=client, enable_reflection=False)
            self.assertIn('saved', output)
            self.assertTrue((Path(tmp)/'leads.jsonl').exists())
            self.assertEqual(len(mock.requests), 2)
            result_messages = mock.requests[1]['messages']
            self.assertTrue(any(m['role']=='tool' and 'success' in m['content']
                                for m in result_messages))

    def test_feedback_route_forces_model_function(self):
        call = SimpleNamespace(
            id='fake-call-2',
            function=SimpleNamespace(name='record_feedback',
                arguments=json.dumps({'question': 'Will you launch Japan group tours next year?'})),
        )
        mock = FakeCompletions([
            mock_response(calls=[call]),
            mock_response('That is not confirmed. I have recorded your question.'),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            output = core.run_agent('Will you launch Japan group tours next year?',
                                    client=client, enable_reflection=False)
            self.assertIn('recorded', output)
            self.assertTrue((Path(tmp)/'feedback.jsonl').exists())
            self.assertEqual(mock.requests[0]['tool_choice']['function']['name'],
                             'record_feedback')

    def test_lead_contact_details_force_lead_tool(self):
        # Regression: a model replied "I've logged your inquiry" without calling the tool
        history = [
            {'role': 'user', 'content': "I'd like to book a consultation for a honeymoon to Bali in June."},
            {'role': 'assistant', 'content': 'Could you please share your name and email address?'},
        ]
        call = SimpleNamespace(
            id='fake-call-3',
            function=SimpleNamespace(name='record_customer_interest',
                arguments=json.dumps({'email': 'dana.test@example.com', 'name': 'Dana Test',
                                      'message': 'Bali honeymoon in June, two travelers'})),
        )
        mock = FakeCompletions([mock_response(calls=[call]), mock_response('Thanks, Dana! Your inquiry is saved.')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            core.run_agent("I'm Dana Test, dana.test@example.com, two travelers.", history,
                           client=client, enable_reflection=False)
            self.assertEqual(mock.requests[0]['tool_choice']['function']['name'], 'record_customer_interest')
            self.assertIn('dana.test@example.com', (Path(tmp)/'leads.jsonl').read_text())

    def test_lead_without_email_lets_model_ask_for_details(self):
        mock = FakeCompletions([mock_response('Happy to help! Could you share your name and email?')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            core.run_agent("I'd like to book a consultation for a honeymoon.", client=client, enable_reflection=False)
            self.assertEqual(mock.requests[0]['tool_choice'], 'auto')

    def test_empty_provider_response_is_retried(self):
        # OpenRouter can answer HTTP 200 with an error body and no choices
        empty = SimpleNamespace(choices=None, error={'message': 'upstream overloaded'})
        mock = FakeCompletions([empty, mock_response('Bali Bliss is $1,850 per person.')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with patch.object(core.time, 'sleep') as sleep:
            output = core.run_agent('What does Bali Bliss cost?', client=client, enable_reflection=False)
        self.assertIn('1,850', output)
        self.assertEqual(len(mock.requests), 2)
        sleep.assert_called_once()

    def _bedrock(self, responses):
        client = SimpleNamespace(requests=[])

        def converse(**kwargs):
            client.requests.append(kwargs)
            return responses.pop(0)
        client.converse = converse
        return client

    @staticmethod
    def _text(text):
        return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn"}

    @staticmethod
    def _tool_use(name, args):
        return {"output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "tu-1", "name": name, "input": args}}]}}, "stopReason": "tool_use"}

    def test_bedrock_feedback_forces_tool_then_answers(self):
        client = self._bedrock([
            self._tool_use("record_feedback", {"question": "What is your cancellation policy?"}),
            self._text("That isn't defined yet; I've recorded your question."),
        ])
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)), \
             patch.object(core, '_bedrock_client', return_value=client):
            meta = core.run_agent_with_metadata('What is your cancellation policy?', provider='bedrock',
                                                enable_reflection=False)
            self.assertTrue((Path(tmp)/'feedback.jsonl').exists())
        self.assertEqual(client.requests[0]['toolConfig']['toolChoice'], {'tool': {'name': 'record_feedback'}})
        self.assertEqual(client.requests[1]['toolConfig']['toolChoice'], {'auto': {}})  # no forced loop
        tool_result = client.requests[1]['messages'][-1]['content'][0]['toolResult']
        self.assertEqual(tool_result['content'][0]['json']['status'], 'success')
        self.assertEqual([t['name'] for t in meta['tools_called']], ['record_feedback'])
        self.assertIn('recorded', meta['reply'])

    def test_bedrock_lead_with_email_forces_lead_tool(self):
        history = [{'role': 'user', 'content': 'I want to book a Bali honeymoon consultation'},
                   {'role': 'assistant', 'content': 'May I have your name and email?'}]
        client = self._bedrock([
            self._tool_use("record_customer_interest",
                           {"email": "dana@example.com", "name": "Dana", "message": "Bali honeymoon"}),
            self._text("Thanks Dana, your inquiry is saved; nothing is booked."),
        ])
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)), \
             patch.object(core, '_bedrock_client', return_value=client):
            core.run_agent('Dana, dana@example.com', history, provider='bedrock', enable_reflection=False)
            self.assertIn('dana@example.com', (Path(tmp)/'leads.jsonl').read_text())
        self.assertEqual(client.requests[0]['toolConfig']['toolChoice'], {'tool': {'name': 'record_customer_interest'}})

    def test_bedrock_general_question_is_auto_and_single_call(self):
        client = self._bedrock([self._text('Bali Bliss is $1,850 per person.')])
        with patch.object(core, '_bedrock_client', return_value=client):
            reply = core.run_agent('What does Bali Bliss cost?', provider='bedrock', enable_reflection=False)
        self.assertIn('1,850', reply)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]['toolConfig']['toolChoice'], {'auto': {}})

    def test_bedrock_messages_alternate_and_start_with_user(self):
        history = [{'role': 'assistant', 'content': 'Welcome!'},
                   {'role': 'user', 'content': 'Hi'}, {'role': 'user', 'content': 'Bali?'}]
        turns = core._bedrock_messages('And Rome?', history)
        self.assertEqual([t['role'] for t in turns], ['user'])
        self.assertIn('Bali?', turns[0]['content'][0]['text'])
        self.assertIn('And Rome?', turns[0]['content'][0]['text'])

    def test_reflection_strips_revised_reply_label(self):
        client = self._bedrock([self._text("**REVISED REPLY:**\n\nWanderWise doesn't book flights.")])
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)), \
             patch.object(core, '_bedrock_client', return_value=client):
            revised = core.reflect_on_answer("We book flights!", "general_question", provider="bedrock")
        self.assertEqual(revised, "WanderWise doesn't book flights.")

    def test_reflection_builds_its_own_openrouter_client(self):
        # Regression: without a client argument the OpenRouter/OpenAI reflection silently returned the draft
        mock = FakeCompletions([mock_response("WanderWise doesn't book flights; refund terms aren't published.")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)), \
             patch.object(core, '_openai_client', return_value=client):
            revised = core.reflect_on_answer("We book flights! Full refunds anytime.", "general_question",
                                             provider="openrouter")
        self.assertIn("doesn't book flights", revised)
        self.assertEqual(len(mock.requests), 1)

    def test_reflection_tool_markup_falls_back_to_draft(self):
        markup = '<dots_function_call> <invoke name="record_feedback"> </invoke> </dots_function_call>'
        mock = FakeCompletions([mock_response(markup)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)), \
             patch.object(core, '_openai_client', return_value=client):
            revised = core.reflect_on_answer("Original draft.", "feedback_unanswered", provider="openrouter")
        self.assertEqual(revised, "Original draft.")

    def test_junk_lead_args_are_repaired_from_customer_words(self):
        # Regression: a model sent ': ' for every field although the customer gave name, email and trip
        history = [{'role': 'user', 'content': "I'd like to book a consultation for a honeymoon to Bali in June."},
                   {'role': 'assistant', 'content': 'Could you share your name, email and a short trip note?'}]
        call = SimpleNamespace(id='c', function=SimpleNamespace(
            name='record_customer_interest', arguments=json.dumps({'email': ': ', 'name': ': ', 'message': ': '})))
        mock = FakeCompletions([mock_response(calls=[call]), mock_response('Thanks Jad, your inquiry is saved.')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            core.run_agent("I'm Jad Elmoghraby, jne20@mail.aub.edu", history, client=client, enable_reflection=False)
            lead = json.loads((Path(tmp)/'leads.jsonl').read_text().splitlines()[0])
        self.assertEqual(lead['name'], 'Jad Elmoghraby')
        self.assertEqual(lead['email'], 'jne20@mail.aub.edu')
        self.assertIn('honeymoon to Bali in June', lead['message'])

    def test_lead_repair_never_invents_a_missing_name(self):
        repaired = core._repair_lead_args({'email': '', 'name': '', 'message': ''},
                                          'my email is someone@example.com',
                                          [{'role': 'user', 'content': 'I want a Rome trip consultation'}])
        self.assertEqual(repaired['email'], 'someone@example.com')
        self.assertEqual(repaired['name'], '')  # not in the conversation: the tool must still refuse
        result = core.record_customer_interest(**repaired)
        self.assertEqual(result['status'], 'error')

    def test_router_runs_required_lead_call_when_model_skips_it(self):
        # Regression: with tool_choice forced, a model still answered with no tool call and no text
        history = [{'role': 'user', 'content': "I'd like to book a consultation for a honeymoon to Bali in June."},
                   {'role': 'assistant', 'content': 'Could you share your name and email?'}]
        mock = FakeCompletions([mock_response(''), mock_response('Thanks Jad, your inquiry is saved.')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=mock))
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            meta = core.run_agent_with_metadata("I'm Jad Elmoghraby, jne20@mail.aub.edu", history,
                                                client=client, enable_reflection=False)
            lead = json.loads((Path(tmp)/'leads.jsonl').read_text().splitlines()[0])
        self.assertEqual((lead['name'], lead['email']), ('Jad Elmoghraby', 'jne20@mail.aub.edu'))
        self.assertEqual([t['name'] for t in meta['tools_called']], ['record_customer_interest'])
        sent = mock.requests[1]['messages']
        self.assertTrue(any(m['role'] == 'tool' and 'success' in m['content'] for m in sent))

    def test_tool_rejects_punctuation_only_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'LOG_DIR', Path(tmp)):
            result = core.record_customer_interest('jne20@mail.aub.edu', ', ', 'Bali honeymoon')
            self.assertEqual(result['status'], 'error')
            self.assertFalse((Path(tmp)/'leads.jsonl').exists())

    def test_lead_note_drops_contact_fragments(self):
        repaired = core._repair_lead_args({'email': ':', 'name': ':', 'message': ':'},
                                          "I'm Jad Elmoghraby, jne20@mail.aub.edu",
                                          [{'role': 'user', 'content': "I'd like a consultation for a honeymoon to Bali in June."}])
        self.assertEqual(repaired['message'], "I'd like a consultation for a honeymoon to Bali in June")

    def test_valid_lead_args_are_left_unchanged(self):
        args = {'email': 'dana@example.com', 'name': 'Dana Saleh', 'message': 'Bali honeymoon, 2 travelers'}
        self.assertEqual(core._repair_lead_args(args, 'anything', []), args)

    def test_router_unanswerable_domain_inquiries(self):
        """Test unanswerable topics from FAQ and business policies route to feedback."""
        self.assertEqual(core.route_message('What is your cancellation policy?'), 'feedback_unanswered')
        self.assertEqual(core.route_message('Can I get a refund if I cancel?'), 'feedback_unanswered')
        self.assertEqual(core.route_message('Do you provide travel insurance?'), 'feedback_unanswered')
        self.assertEqual(core.route_message('Can you issue my tourist visa?'), 'feedback_unanswered')

    def test_mcp_schemas_structure(self):
        """Verify MCP tool definitions contain inputSchema and outputSchema per specification."""
        self.assertEqual(len(core.MCP_TOOL_DEFINITIONS), 2)
        for tool in core.MCP_TOOL_DEFINITIONS:
            self.assertIn('name', tool)
            self.assertIn('description', tool)
            self.assertIn('inputSchema', tool)
            self.assertIn('outputSchema', tool)
            self.assertEqual(tool['inputSchema']['type'], 'object')
            self.assertEqual(tool['outputSchema']['type'], 'object')
            self.assertIn('required', tool['inputSchema'])
            self.assertIn('required', tool['outputSchema'])

    def test_provider_detection(self):
        """Verify provider auto-detection logic."""
        with patch.dict('os.environ', {'LLM_PROVIDER': 'gemini'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'gemini')
        with patch.dict('os.environ', {'LLM_PROVIDER': 'openrouter'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'openrouter')
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'fake-key'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'gemini')
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake-key'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'openrouter')
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'openai')
        with patch.dict('os.environ', {'AWS_BEARER_TOKEN_BEDROCK': 'fake-token'}, clear=True):
            self.assertEqual(core.get_active_provider(), 'bedrock')
            self.assertEqual(core.get_active_model(), 'us.anthropic.claude-haiku-4-5-20251001-v1:0')


if __name__ == '__main__':
    unittest.main()

