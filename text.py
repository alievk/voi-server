from typing import Generator, Iterator
from nltk.tokenize import sent_tokenize
import nltk


class SentenceStream:
    def __init__(self, generator, preprocessor=None):
        nltk.download('punkt', quiet=True)
        self.generator = generator
        self.preprocessor = preprocessor
        self.buffer = ""

    def __iter__(self):
        for chunk in self.generator:    
            if self.preprocessor:
                chunk = self.preprocessor(chunk)

            if not chunk:
                continue

            self.buffer += chunk
            sentences = sent_tokenize(self.buffer)
            
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip():
                        yield sent.strip()
                self.buffer = sentences[-1]
        
        if self.buffer.strip():
            for sent in sent_tokenize(self.buffer):
                if sent.strip():
                    yield sent.strip()
            self.buffer = ""