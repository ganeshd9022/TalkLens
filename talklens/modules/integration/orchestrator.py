"""
Multimodal Integration Layer
──────────────────────────────
Routes messages between Vision, Sign, and Speech modules.
Optionally queries an LLM (Groq/OpenAI) for contextual responses.

Two-way communication contract:
  • Blind user speaks  → STT → text displayed for deaf user
  • Deaf user signs    → sign text → TTS plays for blind user
  • "What's in front?" → vision description → LLM → spoken answer
"""
from __future__ import annotations

import time
from typing import List, Optional
from loguru import logger
from config.settings import config


class ConversationMemory:
    """Rolling context window for LLM conversations."""

    def __init__(self, max_turns: int = 10) -> None:
        self._history: List[dict] = []
        self._max = max_turns

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        if len(self._history) > self._max * 2:
            self._history = self._history[-self._max * 2:]

    def get(self) -> List[dict]:
        return self._history.copy()

    def clear(self) -> None:
        self._history.clear()


class LLMClient:
    """Thin wrapper around Groq/OpenAI for contextual scene understanding."""

    SYSTEM_PROMPT = (
        "You are TalkLens, an AI assistant for the visually impaired. "
        "You will receive a description of the current camera scene and a question from the user. "
        "Answer the user's question directly and concisely (1 sentence) using ONLY the provided scene description. "
        "If the scene doesn't contain the answer, say 'I cannot see that in the current scene'. "
        "Do not invent or hallucinate objects not mentioned in the scene."
    )

    def __init__(self) -> None:
        self._cfg    = config.integration
        self._client = None

    def _init_client(self) -> None:
        provider = self._cfg.llm_provider
        import os
        
        try:
            if provider == "groq" and os.getenv("GROQ_API_KEY"):
                from groq import Groq
                self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            elif (provider == "openai" or "gpt" in self._cfg.llm_model) and os.getenv("OPENAI_API_KEY"):
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif provider == "gemini" and os.getenv("GOOGLE_API_KEY"):
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                # Use standard chat-based model
                self._client = genai.GenerativeModel(
                    model_name=self._cfg.llm_model,
                    system_instruction=self.SYSTEM_PROMPT
                )
            
            if self._client:
                logger.info(f"LLM client initialised: {provider}/{self._cfg.llm_model}")
        except Exception as e:
            logger.warning(f"LLM init failed ({e}).")

    def query(self, user_msg: str, history: List[dict]) -> str:
        if not self._client and self._cfg.enable_llm:
            self._init_client()
            
        if not self._client:
            return ""

        try:
            # Pop the raw question from history since we are sending a formatted context prompt
            clean_history = history[:-1] if history and history[-1]["role"] == "user" else history

            # Handle Google Gemini differently
            if self._cfg.llm_provider == "gemini":
                full_prompt = ""
                if len(clean_history) > 0:
                    full_prompt += "Previous Context:\n"
                    for h in clean_history[-3:]:
                        full_prompt += f"{h['role'].upper()}: {h['content']}\n"
                
                full_prompt += f"\n{user_msg}"
                
                chat = self._client.start_chat(history=[])
                resp = chat.send_message(full_prompt)
                return resp.text.strip()

            # Standard OpenAI-compatible call with timeout to prevent UI hang
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            messages += clean_history[-10:]
            messages.append({"role": "user", "content": user_msg})
            
            resp = self._client.chat.completions.create(
                model    = self._cfg.llm_model,
                messages = messages,
                max_tokens = 256,
                temperature = 0.6,
                timeout = 10.0, # 10 second timeout to prevent camera freeze
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return f"AI error: {str(e)[:50]}"


class TalkLensOrchestrator:
    """
    Central coordinator — connects all modules and drives the two-way
    communication flow.
    """

    def __init__(self) -> None:
        self._cfg    = config.integration
        self._memory = ConversationMemory(max_turns=self._cfg.context_window)
        self._llm    = LLMClient()
        self._last_scene_description = ""
        self._last_sign_text = ""

    # ── Vision Assistant Flow ──────────────────────────────────────────────────

    def handle_vision_update(
        self,
        detections: list,
        scene_description: str,
        spatial_context: str,
    ) -> str:
        """
        Called every time new detections arrive.
        Returns the response text (may be LLM-enriched).
        """
        self._last_scene_description = scene_description
        if not detections:
            return ""

        if self._cfg.enable_llm:
            prompt = (
                f"Describe this scene naturally: {scene_description} at {spatial_context}"
            )
            response = self._llm.query(prompt, self._memory.get())
            if response:
                self._memory.add("assistant", response)
                return response
 
        return scene_description  # default (already starts with 'I see:')

    def handle_user_question(self, question: str) -> str:
        """
        Blind user speaks a question (e.g. "What is in front of me?").
        Answers using scene context + optional LLM.
        """
        context = f"SCENE: {self._last_scene_description}\nQUESTION: {question}"
        self._memory.add("user", question)
        
        # We let the LLM client decide if it can answer (auto-inits if key provided)
        answer = self._llm.query(context, self._memory.get())
        
        if not answer:
            # Fallback for when LLM is disabled or key is missing
            answer = f"Based on the camera, {self._last_scene_description}"
            
        self._memory.add("assistant", answer)
        return answer

    # ── Sign Language → Blind User Flow ───────────────────────────────────────

    def handle_sign_text(self, sign_text: str) -> str:
        """
        Deaf user finishes signing a word/sentence.
        Converts to speech-friendly response.
        """
        self._last_sign_text = sign_text
        self._memory.add("user", f"[SIGNED]: {sign_text}")
        if self._cfg.enable_llm and self._llm._client:
            response = self._llm.query(sign_text, self._memory.get())
            self._memory.add("assistant", response)
            return response
        return sign_text

    # ── Blind User Speech → Deaf User Flow ────────────────────────────────────

    def handle_speech_transcript(self, transcript: str) -> str:
        """
        Blind user speaks → returns polished text to display for deaf user
        (and optionally gets an LLM-improved response).
        """
        self._memory.add("user", f"[SPOKEN]: {transcript}")
        return transcript   # display verbatim; LLM enhancement is future work

    def clear_context(self) -> None:
        self._memory.clear()
