
from bs4 import BeautifulSoup
import spacy
import polars as pl

class BagOfWordsExtractor:

    def __init__(
        self,
        permitted_words: list[str],
    ):
        self.nlp = spacy.load('en_core_web_trf', disable=['parser'])  # Largest, slowest, most accurate model
        self.permitted_words = permitted_words

    def extract_bag_of_words(self, html: str) -> dict[str, int]:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=' ', strip=True)
        spacy_doc = self.nlp(text)
        bag_of_words = {word: 0 for word in self.permitted_words}
        for token in spacy_doc:
            if token.is_stop:
                continue
            if not token.is_alpha:
                continue
            if len(token.text) < 2:
                continue
            text = token.lemma_.lower()
            if text not in self.permitted_words:
                continue
            bag_of_words[text] = 1
        return bag_of_words

    def format_bag_of_words(
        self,
        bag_of_words: dict[str, int]
    ) -> pl.DataFrame:
        raise NotImplementedError


