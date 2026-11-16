"""Offline checks for the research agent's model configuration."""

import unittest
from unittest import mock

from genai_processors.core import genai_model
from genai_processors.examples.research import agent
from genai_processors.examples.research import interfaces
from google.genai import types


class ResearchModelConfigTest(unittest.TestCase):

  def test_default_stages_use_replacement_with_medium_thinking(self):
    with mock.patch.object(
        genai_model, 'GenaiModel', wraps=genai_model.GenaiModel
    ) as create_model:
      agent.ResearchAgent(api_key='test-key')

    self.assertEqual(create_model.call_count, 3)
    for call in create_model.call_args_list:
      self.assertEqual(call.kwargs['model_name'], 'gemini-3.5-flash-lite')
      config = types.GenerateContentConfig.model_validate(
          call.kwargs['generate_content_config']
      )
      self.assertEqual(config.thinking_config.thinking_level, 'MEDIUM')
      self.assertIsNone(config.thinking_config.thinking_budget)
      self.assertIsNone(config.temperature)

  def test_explicit_older_models_do_not_receive_thinking_levels(self):
    config = interfaces.Config(
        topic_generator_model_name='gemini-2.5-flash-lite',
        topic_researcher_model_name='gemini-2.5-flash-lite',
        research_synthesizer_model_name='gemini-2.5-flash-lite',
    )
    with mock.patch.object(
        genai_model, 'GenaiModel', wraps=genai_model.GenaiModel
    ) as create_model:
      agent.ResearchAgent(api_key='test-key', config=config)

    self.assertEqual(create_model.call_count, 3)
    for call in create_model.call_args_list:
      self.assertEqual(call.kwargs['model_name'], 'gemini-2.5-flash-lite')
      request_config = types.GenerateContentConfig.model_validate(
          call.kwargs['generate_content_config']
      )
      self.assertIsNone(request_config.thinking_config)


if __name__ == '__main__':
  unittest.main()
