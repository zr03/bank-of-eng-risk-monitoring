from llms.base_llm import BaseLLM

class ChunkingLLM(BaseLLM):
    def _make_chunk_prompt(self, text):
        return (
            "Split this portion of a financial transcript into logical sections based on topic, speaker changes, or key points. Return a JSON-like structure.\n\n"
            f"Transcript:\n{text[:2000]}"
        )

    def chunk_transcript(self, text):
        prompt = self._make_chunk_prompt(text)
        return self.invoke(prompt)

    async def chunk_transcript_async(self, text):
        prompt = self._make_chunk_prompt(text)
        return await self.ainvoke(prompt)
