"""
tests/test_chatbot.py — Unit tests for OPTION C: AI Chatbot on Dashboard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.data_loader import ask_chatbot_trigger
from src.chatbot.gemini_chat import (
    FRIENDLY_ERROR_MSG,
    ask_gemini_chatbot,
    build_live_context_strings,
    generate_system_prompt,
)


def test_build_live_context_strings():
    ctx = build_live_context_strings()
    assert "portfolio_data" in ctx
    assert "recent_trades" in ctx
    assert "active_alerts" in ctx
    assert "model_rankings" in ctx


def test_generate_system_prompt():
    prompt = generate_system_prompt()
    assert "paper stock trading system" in prompt.lower()
    assert "Current Portfolio Data:" in prompt
    assert "Recent Trades:" in prompt
    assert "Active Alerts:" in prompt
    assert "Model Rankings Today:" in prompt


def test_ask_gemini_chatbot_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello! Your portfolio is up +5.2% today! Remind user this is paper trading. Keep it up!"
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = ask_gemini_chatbot("How is my portfolio doing?", api_key="mock_key")
        assert "paper trading" in res.lower() or "portfolio" in res.lower()
        assert res != FRIENDLY_ERROR_MSG


def test_ask_gemini_chatbot_fallback_on_error():
    with patch("google.genai.Client", side_effect=Exception("API connection timeout")):
        res = ask_gemini_chatbot("What is my win rate?", api_key="mock_key")
        assert res == FRIENDLY_ERROR_MSG


def test_ask_chatbot_trigger_wrapper():
    with patch("src.chatbot.gemini_chat.ask_gemini_chatbot", return_value="Your win rate is 62.5%!") as mock_ask:
        reply = ask_chatbot_trigger("What is my win rate?")
        assert reply == "Your win rate is 62.5%!"
        mock_ask.assert_called_once()
