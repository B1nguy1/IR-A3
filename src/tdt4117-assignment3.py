import string
import random
import codecs
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist

########
# TASK 1 #
#########

# 1.0
random.seed(123)

# 1.1-1.7

# 1.1
with codecs.open("pg3300.txt", "r", encoding="utf-8") as paragraphs_file:
    paragraphs = paragraphs_file.read()

# 1.2
    partitioned_paragraphs = paragraphs.split("\r\n\r\n")

    paragraphs_without_whitespaces = []
    for partioned_paragraph in partitioned_paragraphs:
        # Remove whitespaces and add to a new list
        paragraphs_without_whitespaces.append(partioned_paragraph.strip())

    paragraphs_without_empty_strings = []
    for p in paragraphs_without_whitespaces:
        # Add paragraph to the list if not empty.
        if p != "":
            paragraphs_without_empty_strings.append(p)

    # 1.3
    paragraphs_without_gutenberg = []
    for paragraph in paragraphs_without_empty_strings:
        filter_word = "Gutenberg".lower()
        if filter_word not in paragraph.lower():
            paragraphs_without_gutenberg.append(paragraph)

    copy_of_paragraphs_list = paragraphs_without_gutenberg[:]

    # 1.4

    paragraphs_not_empty = []
    for paragraph in paragraphs_without_gutenberg:
        single_word = paragraph.split(" ")
        paragraphs_not_empty.append(single_word)

    # 1.5
    translation_table = str.maketrans("", "", string.punctuation+"\n\r\t")
    list1 = [[word.lower().translate(word) for word in paragraph]
             for paragraph in paragraphs_not_empty]

    # 1.6
    stemmer = PorterStemmer()
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            list1[i][j] = stemmer.stem(list1[i][j])

    # 1.7
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            freqDist = FreqDist(list1[i][j])

  #  print(list1[:10])
    ########
    # TASK 2 #
    #########

    # 2.0
    dictionary = Dictionary(list1)
    # 2.1
    with codecs.open("common-english-words.txt", "r", encoding="utf-8") as english_words:

        stop_words = english_words.read().split(",")
        stop_ids = []
        for stopword in stop_words:
            if stopword in dictionary.values():
                stop_ids.append(dictionary.token2id[stopword])
        dictionary.filter_tokens(stop_ids)

    # 2.2

        bow_corpus = [dictionary.doc2bow(paragraph)
                      for paragraph in list1]
    ########
    # TASK 3 #
    #########

    # 3.1
        tfidf_model = TfidfModel(bow_corpus)

    # 3.2
        tfidf_corpus = tfidf_model[bow_corpus]

    # 3.3
        tfidf_matrixSimilarity = MatrixSimilarity(tfidf_corpus)
       # print(tfidf_matrixSimilarity)

    # 3.4
        lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
        lsi_corpus = lsi_model[tfidf_corpus]
        # print(lsi_corpus)
        lsi_matrixSimilarity = MatrixSimilarity(lsi_corpus)
        # print(lsi_model.show_topics(num_topics=3))

    ########
    # TASK 4 #
    #########

    # 4.1

        def preprocessing(doc):
            stemmer = PorterStemmer()
            translation_table = str.maketrans(
                "", "", string.punctuation+"\n\r\t")
            lowercase_and_without_punctuation = [
                word.lower().translate(translation_table) for word in doc]
            stemmed_documents = [stemmer.stem(
                word) for word in lowercase_and_without_punctuation]
            return stemmed_documents

        query = "What is the function of money?".split(" ")
        q = preprocessing(query)
        q = dictionary.doc2bow(q)

    # 4.2
        tfidf_Query = tfidf_model[q]

    # 4.3
        doc2similarity = enumerate(tfidf_matrixSimilarity[tfidf_Query])
        top3Paragraphs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

        for doc in top3Paragraphs:
            print("TFIDF:")
            print(f'\n[paragraph {doc[0]}]')
            document = copy_of_paragraphs_list[doc[0]]
            print("\n".join(document.split("\n")[:5]))

        # 4.4
        lsi_query = lsi_model[tfidf_Query]
        top3Topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
        topics = lsi_model.show_topics()[:3]

        doc2similarity = enumerate(lsi_matrixSimilarity[lsi_query])
        docs2 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

        for d in docs2:
            print("LSI:")
            print(f'\n[paragraph {d[0]}]')
            test2 = copy_of_paragraphs_list[d[0]]
            print("\n".join(test2.split("\n")[:5]))
