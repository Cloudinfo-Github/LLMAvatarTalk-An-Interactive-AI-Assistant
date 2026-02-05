from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationChain
from langchain_community.chat_models import ChatOllama
from config import LANGUAGE, OLLAMA_BASE_URL, OLLAMA_MODEL
import re


class LLMService:
    def __init__(self, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, k=5):
        self.language = LANGUAGE
        self.model = model
        self.base_url = base_url
        self.k = k
        self.prompt = self.setup_prompt()
        self.llm = ChatOllama(model=self.model, base_url=self.base_url)
        self.conversation = ConversationChain(
            prompt=self.prompt,
            llm=self.llm,
            verbose=False,
            memory=ConversationBufferWindowMemory(k=self.k),
        )

    def setup_prompt(self):
        if self.language == 'zh-CN':
            template = """Always response in Chinese(汉字), not English

            Current conversation:
            {history}
            Human: {input}
            AI Assistant:"""
        else:
            template = """The following is a friendly conversation between a human and an AI. AI will reply within 100 words. If the AI does not know the answer to a question, it truthfully says it does not know.

            Current conversation:
            {history}
            Human: {input}
            AI Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"], template=template)
        return prompt

    def split_text_by_period(self, text, max_length=200):
        """
        Splits the text into chunks that are less than or equal to the max_length,
        cutting at sentence boundaries denoted by punctuation such as periods, exclamation marks, or colons.
        This function ensures that the text is split into readable chunks without exceeding the specified maximum length.

        Args:
            text (str): The full text to split.
            max_length (int): Maximum length of each chunk.

        Returns:
            List[str]: A list of text chunks that adhere to the maximum length and maintain logical sentence boundaries.
        """
        # Define the pattern for splitting which includes major sentence-ending punctuations
        pattern = re.compile(r'([。！,]+)')
        sentences = pattern.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Add back the punctuation to the sentence which was removed by splitting
            if pattern.match(sentence):
                current_chunk += sentence
            else:
                # Check if adding this sentence would exceed the max_length
                if len(current_chunk) + len(sentence) <= max_length:
                    current_chunk += sentence
                else:
                    # If the current chunk is not empty, append it and start a new chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        # Append the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def invoke_conversation(self, user_input):
        result = self.conversation.invoke(user_input)
        response_chunks = self.split_text_by_period(result['response'])
        return response_chunks
