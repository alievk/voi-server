from nltk.tokenize import sent_tokenize
import nltk
import re


NARRATOR_MARKER = "@"


def split_text_into_chunks(text, avg_text_len=60):
    chunks = []
    buffer = ""
    current_len = 0
    
    for sent in SentenceStream(text):
        sent_len = len(sent)
        
        if not buffer:
            buffer = sent
            current_len = sent_len
            continue
            
        new_len = current_len + len(sent) + 1  # +1 for space
        
        if abs(new_len - avg_text_len) < abs(current_len - avg_text_len):
            buffer += ' ' + sent
            current_len = new_len
        else:
            chunks.append(buffer)
            buffer = sent
            current_len = sent_len
            
    if buffer:
        chunks.append(buffer)
    return chunks


def normalize_text(text):
    text = text.strip()
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace("*", '')
    return text


def split_text_into_speech_segments(text, avg_text_len=60, narrator_marker=NARRATOR_MARKER):
    """
    Return list of speech segments with text and role.
    Role is either "narrator" or "character".
    Narrator is the text between narrator_marker, the rest is character.
    """
    def process_segment(segment, role):
        if not segment:
            return []

        segment = normalize_text(segment)
        
        parts = []
        for segment in split_text_into_chunks(segment, avg_text_len):
            if not any(c.isalpha() for c in segment):
                continue
                
            segment = segment.rstrip(',')
            if not segment.endswith(('...', '.', '!', '?')):
                segment += '.'
            parts.append({"text": segment, "role": role})
        return parts

    parts = []
    last_end = 0
    
    pattern = rf"\{narrator_marker}([^{narrator_marker}]+)\{narrator_marker}[.,!?]?"
    for match in re.finditer(pattern, text):
        # Process character text before narrator text
        parts.extend(process_segment(text[last_end:match.start()], "character"))
        # Process narrator text 
        parts.extend(process_segment(match.group(1), "narrator"))
        last_end = match.end()
    
    # Process remaining character text
    parts.extend(process_segment(text[last_end:], "character"))
    
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