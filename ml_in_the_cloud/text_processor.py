import re
from typing import List, Optional, Dict
from collections import Counter

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


class TextPreprocessor:
    def __init__(self,
                 long_token_threshold: int = 20,
                 frequency_threshold: int = 2,
                 extra_stopwords: Optional[List[str]] = None,
                 pos_stoplist: Optional[List[str]] = None):
        """
        Initialise la pipeline NLP.
        """
        self.long_token_threshold = long_token_threshold
        self.frequency_threshold = frequency_threshold
        self.stop_words = set(stopwords.words('english'))
        if extra_stopwords:
            self.stop_words.update(extra_stopwords)

        # POS stoplist : ex. ['DT', 'CC']
        self.pos_stoplist = set(pos_stoplist) if pos_stoplist else set()

        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoyage de base : suppression espaces multiples, ponctuation superflue.
        """
        text = re.sub(r'\s+', ' ', text)  # espaces multiples
        text = re.sub(r'[^\w\s.,!?]', '', text)  # caractères spéciaux sauf ponctuation basique
        text = text.strip()
        return text

    @staticmethod
    def get_wordnet_pos(tag: str) -> str:
        """
        Conversion des tags POS en format WordNet.
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def process_sentence(self, sentence: str) -> List[str]:
        """
        Nettoie une phrase : tokenisation, lemmatisation, filtrage.
        """
        tokens = word_tokenize(sentence)
        tokens = [t.lower() for t in tokens if len(t) <= self.long_token_threshold]

        # POS tagging
        pos_tags = pos_tag(tokens)

        # Lemmatisation + filtre POS stoplist + stopwords
        cleaned_tokens = []
        for token, tag in pos_tags:
            if token in self.stop_words:
                continue
            if tag in self.pos_stoplist:
                continue
            if re.fullmatch(r'\W+', token):  # depunctuation
                continue
            lemma = self.lemmatizer.lemmatize(token, self.get_wordnet_pos(tag))
            cleaned_tokens.append(lemma)

        return cleaned_tokens

    def __call__(self, text: str) -> List[List[str]]:
        """
        Pipeline principale sur un texte complet.
        Retourne une liste de phrases tokenisées nettoyées.
        """
        text = self.clean_text(text)
        sentences = sent_tokenize(text)

        processed_sentences = []
        word_counter = Counter()

        # Premier passage : tokenisation et comptage
        intermediate_sentences = []
        for sentence in sentences:
            tokens = self.process_sentence(sentence)
            word_counter.update(tokens)
            intermediate_sentences.append(tokens)

        # Fréquence filter
        for tokens in intermediate_sentences:
            tokens = [t for t in tokens if word_counter[t] >= self.frequency_threshold]
            if tokens:
                processed_sentences.append(tokens)

        return processed_sentences
