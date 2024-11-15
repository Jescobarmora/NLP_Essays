from gensim import corpora, models

def train_lda_model(corpus, dictionary, num_topics=8):
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    return lda_model

def get_topic_distribution(model, corpus):
    topic_distribution = []
    for doc in corpus:
        topics = model.get_document_topics(doc)
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        topic_distribution.append(topics[0][0])
    return topic_distribution

def get_topic_probabilities(model, corpus, num_topics):
    topic_probabilities = []
    for doc in corpus:
        topics = model.get_document_topics(doc, minimum_probability=0)
        topic_dist = [0] * num_topics
        for topic_num, prob in topics:
            topic_dist[topic_num] = prob
        topic_probabilities.append(topic_dist)
    return topic_probabilities