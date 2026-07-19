import re
from typing import List,Union
from pathlib import Path 


class TextTokenizer:
    def __init__(self,lowercase:bool = True, remove_non_alphanumeric : bool = True):
        self.lowercase = lowercase
        self.remove_non_alphanumeric = remove_non_alphanumeric
        self.clean_pattern = re.compile(r'[^a-z0-9\s]')
        self.sentence_pattern = re.compile(r'[.!?]+')
    def clean_text(self,text:str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.remove_non_alphanumeric:
            text = self.clean_pattern.sub('',text)
        
        return text
    
    def tokenize(self,source:Union[str,Path]) -> List[str]:
        if isinstance(source,(str,Path)) and Path(source).is_file():
            with open(source, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            raw_text = str(source)



        corpus = []
        sentences = self.sentence_pattern.split(raw_text)

        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            tokens = cleaned.split()

            if tokens:
                corpus.append(tokens)
        return corpus 


    
    