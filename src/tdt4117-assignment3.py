import string
import random
import codecs
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist

'''
See comments for following tasks "# TASK 1,2,3,4 #", and the following comments for subtasks I have solved. 
'''
########
# TASK 1 #
#########

# 1.0
random.seed(123)


# 1.1
with codecs.open("pg3300.txt", "r", encoding="utf-8") as paragraphs_file:
    paragraphs = paragraphs_file.read()

# 1.2
    partitioned_paragraphs = paragraphs.split("\r\n\r\n")

    paragraphs_without_whitespaces = []
    for partioned_paragraph in partitioned_paragraphs:
        # add each paragraph in partioned paragraphs to a new list without whitespaces
        paragraphs_without_whitespaces.append(partioned_paragraph.strip())

    paragraphs_without_empty_strings = []
    for p in paragraphs_without_whitespaces:
        if p != "":
            paragraphs_without_empty_strings.append(p)

    # 1.3
    paragraphs_without_gutenberg = []
    for paragraph in paragraphs_without_empty_strings:
        filter_word = "Gutenberg".lower()
        if filter_word not in paragraph.lower():
            paragraphs_without_gutenberg.append(paragraph)

    # using same list to convert to lower case
    paragraphs_without_gutenberg = [x.lower()
                                    for x in paragraphs_without_gutenberg]
    # Create a copy of the list for use in part 4
    copy_of_paragraphs_list = paragraphs_without_gutenberg[:]

    # 1.4
    word_list = []
    for paragraph in paragraphs_without_gutenberg:
        single_word = paragraph.split(" ")
        word_list.append(single_word)

    # Removes empty string in the list of words
    word_list = [[i for i in item if i != '']
                 for item in word_list]

    # 1.5
    translation_table = str.maketrans('', '', string.punctuation+"\n\r\t")
    paragraph_list = [[word.lower().translate(translation_table) for word in paragraph]
                      for paragraph in word_list]

    # 1.6
    stemmer = PorterStemmer()
    for i in range(len(paragraph_list)):
        for j in range(len(paragraph_list[i])):
            paragraph_list[i][j] = stemmer.stem(paragraph_list[i][j])

    # 1.7
    for i in range(len(paragraph_list)):
        for j in range(len(paragraph_list[i])):
            freqDist = FreqDist(paragraph_list[i][j])

    ########
    # TASK 2 #
    #########

    # 2.0
    dictionary = Dictionary(paragraph_list)

    # 2.1
    with open("common-english-words.txt", "r", encoding="utf-8") as english_words:

        stop_words = english_words.read().split(",")
        stop_ids = []
        for stopword in stop_words:
            if stopword in dictionary.values():
                stop_ids.append(dictionary.token2id[stopword])
        dictionary.filter_tokens(stop_ids)

    # 2.2

        bow_corpus = [dictionary.doc2bow(paragraph)
                      for paragraph in paragraph_list]
    ########
    # TASK 3 #
    #########

    # 3.1
        tfidf_model = TfidfModel(bow_corpus)

    # 3.2
        tfidf_corpus = tfidf_model[bow_corpus]

    # 3.3
        tfidf_matrixSimilarity = MatrixSimilarity(bow_corpus)

    # 3.4
        lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)

        lsi_corpus = lsi_model[tfidf_corpus]

        lsi_matrixSimilarity = MatrixSimilarity(lsi_corpus)

    # 3.5
        print(f"Topic 1: {lsi_model.show_topic(0)}")
        print("\n")
        print(f"Topic 2: {lsi_model.show_topic(1)}")
        print("\n")
        print(f"Topic 3: {lsi_model.show_topic(2)}")

    ########
    # TASK 4 #
    #########

    # 4.1
        '''
        A function that removes punctuation and stems the query "q" 
        '''
        def preprocessing(q):
            stemmer = PorterStemmer()
            translation_table = str.maketrans(
                '', '', string.punctuation+"\n\r\t")
            lowercase_and_without_punctuation = [
                word.lower().translate(translation_table) for word in q]
            stemmed_documents = [stemmer.stem(
                word) for word in lowercase_and_without_punctuation]
            return stemmed_documents

        # splits the query into a list containing each word from the query sentence
        query = "What is the function of money?".split(" ")
        # runs the function preprocessing
        q = preprocessing(query)
        # converts to a BOW representation
        q = dictionary.doc2bow(q)

    # 4.2
        tfidf_Query = tfidf_model[q]

        for i in tfidf_Query:
            print("index:", i[0], "word:", dictionary.get(
                i[0], i[1]), "influenc:", i[1])

    # 4.3
        doc2similarity = enumerate(tfidf_matrixSimilarity[tfidf_Query])
        top3Paragraphs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

        for p in top3Paragraphs:
            print(f'\n[paragraph {p[0]}]')
            document = copy_of_paragraphs_list[p[0]]
            print()
           # print("\n".join(document.split("\n")[:5]))

        '''
        [paragraph 23]
           chapter iv.
            of the origin and use of money.

        [paragraph 73]
            chapter ii.
            of money, considered as a particular
            branch of the general stock of the society, or of the expense of maintaining
            the national capital.
        
        [paragraph 101]
            Bar or ingot gold is received in proportion to its fineness, compared with
        the above foreign gold coin. Upon fine bars the bank gives 340 per mark.
        In general, however, something more is given upon coin of a known
        fineness, than upon gold and silver bars, of which the fineness cannot be
        ascertained but by a process of melting and assaying.
        '''

    # 4.4
        lsi_query = lsi_model[tfidf_Query]
        top3Topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
        for topic in top3Topics:
            print("[", "Topic:", topic[0], "]")
            print(lsi_model.show_topic(topic[0]))

        doc2similarity = enumerate(lsi_matrixSimilarity[lsi_query])
        docs2 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

        for d in docs2:
            print(f'\n[paragraph {d[0]}]')
            test2 = copy_of_paragraphs_list[d[0]]
            print()
            # print("\n".join(test2.split("\n")[:5]))

        '''
        [paragraph 74]
        It has been shown in the First Book, that the price of the greater part of
        commodities resolves itself into three parts, of which one pays the wages
        of the labour, another the profits of the stock, and a third the rent of
        the land which had been employed in producing and bringing them to market:
        that there are, indeed, some commodities of which the price is made up of

        [paragraph 72]
        When the stock which a man possesses is no more than sufficient to
        maintain him for a few days or a few weeks, he seldom thinks of deriving
        any revenue from it. He consumes it as sparingly as he can, and
        endeavours, by his labour, to acquire something which may supply its place
        before it be consumed altogether. His revenue is, in this case, derived

        [paragraph 78]
            The stock which is lent at interest is always considered as a capital by
            the lender. He expects that in due time it is to be restored to him, and
         that, in the mean time, the borrower is to pay him a certain annual rent
        for the use of it. The borrower may use it either as a capital, or as a
        stock reserved for immediate consumption. If he uses it as a capital, he 
        '''

        '''
        LSI and TFIDF models result in different paragraphs. However, the models include
        paragraphs 72-74.
        '''
