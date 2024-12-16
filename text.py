from nltk.tokenize import sent_tokenize
import nltk
import re


def split_text_into_chunks(text, max_chunk_len=60):
    chunks = []
    buffer = ""
    
    for sent in SentenceStream(text):
        if not buffer:
            buffer = sent
            continue
        if len(buffer) + len(sent) <= max_chunk_len:
            buffer += ' ' + sent
        else:
            chunks.append(buffer)
            buffer = sent
            
    if buffer:
        chunks.append(buffer)
    return chunks


def split_text_into_speech_segments(text, max_chunk_len=60):
    """
    Return list of speech segments with text and role.
    Role is either "narrator" or "character".
    Narrator is the text between * and *, the rest is character.
    """
    def process_chunk(chunk, role):
        if not chunk:
            return []
        
        parts = []
        for segment in split_text_into_chunks(chunk.strip(), max_chunk_len):
            if not any(c.isalpha() for c in segment):
                continue
                
            segment = segment.rstrip(',')
            if not segment.endswith(('...', '.', '!', '?')):
                segment += '.'
            parts.append({"text": segment, "role": role})
        return parts

    parts = []
    last_end = 0
    
    for match in re.finditer(r"\*([^*]+)\*", text):
        # Process character text before narrator text
        parts.extend(process_chunk(text[last_end:match.start()], "character"))
        # Process narrator text
        parts.extend(process_chunk(match.group(1), "narrator"))
        last_end = match.end()
    
    # Process remaining character text
    parts.extend(process_chunk(text[last_end:], "character"))
    
    return parts


class SentenceStream:
    def __init__(self, generator, preprocessor=None):
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
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