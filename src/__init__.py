"""Core domain logic for the Sentiment Analysis Chatbot.

This package contains the business-level building blocks of the application:
data ingestion, model training, inference and chatbot response generation.
The top-level entry points are re-exported here for convenience.
"""

from src.model_inference import SentimentPredictor
from src.response_generator import ChatbotResponder

__all__ = ["ChatbotResponder", "SentimentPredictor"]
