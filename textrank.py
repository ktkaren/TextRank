__doc__ = """
This program can extract keywords from text using the textRank algorithm.
The expected input should be a dataframe (.csv) with a text column.
"""

import pandas as pd
import numpy as np
from gensim.summarization.keywords import keywords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess(dataframe, text_column_name, lemmatizer):

    """
    preprocess the text in dataframe

    :param dataframe: pandas DataFrame
    :param text_column_name: string
    :param lemmatizer: nltk WordNetLemmatizer
    :return:
    """

    # join all texts in the text column into one string
    texts = [text for text in dataframe[text_column_name] if (not pd.isnull(text))]
    joined_text = " ".join(texts)

    # pre-process the concat_text by lemmatizing it
    lemmatized_text = " ".join(lemmatize_all_pos(joined_text, lemmatizer))

    return lemmatized_text


def lemmatize_all_pos(text, lemmatizer, tokenize=True):

    """
    lemmatize each word in text and return a list of lemmatized tokens

    :param text: string
    :param lemmatizer: nltk WordNetLemmatizer
    :param tokenize: boolean (whether the input string needs to be tokenized)
    :return: list of strings (the lemmatized tokens)
    """

    if tokenize:
        tokens = word_tokenize(text)
    else:
        tokens = text
    # lemmatize Verb (v), Noun (n), Adverb (r) and Adjective (a)
    lemmatized_tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t, pos='n') for t in lemmatized_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t, pos='r') for t in lemmatized_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t, pos='a') for t in lemmatized_tokens]

    return lemmatized_tokens


def extract_top_n_percent_keywords(lemmatized_text, lemmatizer, top_n_percent):

    """
    extract keywords using textRank algorithm, lemmatize them and return the top n percent keywords

    :param lemmatized_text: list of strings
    :param lemmatizer: nltk WordNetLemmatizer
    :param top_n_percent: int (0 <= top_n_percent <= 100)
    :return: list of strings (top n percent keywords)
    """

    # use gensim.summarization.keywords to extract keywords
    # this function returns [keyword, scores]
    keyword_score_pair = keywords(lemmatized_text, lemmatize=True, scores=True)
    keywords_only = [x[0] for x in keyword_score_pair]

    # lemmatized the keywords
    lemmatized_keywords = lemmatize_all_pos(keywords_only, lemmatizer, tokenize=False)

    # pair the lemmatized keywords with their scores
    scores = [x[1] for x in keyword_score_pair]
    lemmatized_pairs = list(zip(lemmatized_keywords, scores))

    # select top keywords
    # 100 - int(top_n_percent) percentile is the same as top_n_percent
    # e.g. 90 percentile returns the top 10%
    top_percentile = np.percentile(scores, 100 - int(top_n_percent))
    # pair[0] is the keyword and pair[1] is the corresponding score
    top_keywords = [pair[0] for pair in lemmatized_pairs if pair[1] >= top_percentile]

    return top_keywords


if __name__ == '__main__':

    INPUT_FILE_NAME = input("Please provide the input CSV file. ")
    TEXT_COLUMN_NAME = input("Please provide the column name of the text column. ")
    TOP_N_PERCENT = input("Please enter the percentile for top keywords selection. \
    (i.e enter 10 to select the top 10% keywords) ")
    OUTPUT_FILE_NAME = input("Please provide an output CSV file name. (e.g. output.csv)")

    # set-up
    df = pd.read_csv(INPUT_FILE_NAME)
    data = df.copy()
    LEMMATIZER = WordNetLemmatizer()

    # pre-process the text
    lemmatized_text = preprocess(data, TEXT_COLUMN_NAME, LEMMATIZER)

    # extract and select top n percent keywords
    top_keywords = extract_top_n_percent_keywords(lemmatized_text, LEMMATIZER, TOP_N_PERCENT)

    df_lemmatized_pair = pd.DataFrame(top_keywords, columns=['keywords'])
    df_lemmatized_pair.to_csv(OUTPUT_FILE_NAME, index=False)
