import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import sqlite3
import pandas as pd
import json
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

#nltk.download()
#https://www.nltk.org/data.html
#python -m spacy download en_core_web_sm
class Week4:
    
    def __init__(self, db_file):
        self.dateformat = '%Y-%m-%d %H:%M:%S'
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.tables = self.get_tables()
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['would', 'day', 'like', 'today', 'best', 'always', 'amazing', 'bought', 'quick' 'people', 'new', 'fun', 'think', 'know', 'believe',
                                'many', 'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',
                                'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])
        print('Database connection and class initialized \n')

    def get_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = self.cursor.fetchall()
        clean_tables = [tuple[0] for tuple in tables]
        return clean_tables
    
    def close_connection(self): 
        print("\n Closing connection")
        self.cursor.close()

    def __del__(self):
        # Automatically closes db connection when garbage collected
        self.close_connection()

    @staticmethod
    def pretty_print(data):
        print(json.dumps(data, indent=4))

    def exercise1(self, num_of_topics=10, log_results=True):
        '''
        Exercise 4.1 Topics:
        Identify the 10 most popular topics discussed on our platform. Use Latent Dirichlet Allocation (LDA) with the gensim library.
        Answer and explain your queries/calculations below. You may use SQL and/or Python to perform this task. 5 points)
        '''
        # Materials:
        # https://www.geeksforgeeks.org/nlp/topic-modeling-using-latent-dirichlet-allocation-lda/
        # https://github.com/Crowd-Computing-Oulu/soco-exercise-solutions/blob/main/exercise_task_14.py
        data = pd.read_sql_query(f"SELECT content FROM posts", self.conn)
        
        data['content'] = data['content'].apply(self._preprocess_text)
        lemmatizer = WordNetLemmatizer()
        
        bow_list = []
        for _, row in data.iterrows():
            text = row['content']
            tokens = word_tokenize(text.lower())
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            tokens = [t for t in tokens if len(t) > 2]
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
            if len(tokens) > 0:
                bow_list.append(tokens)

        dictionary = Dictionary(bow_list)
        dictionary.filter_extremes(no_below=2, no_above=0.3)
        corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

        lda = LdaModel(corpus, num_topics=num_of_topics, id2word=dictionary, passes=10, random_state=2)
        coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        topics = lda.print_topics(num_words=5)

        if log_results:
            print(f'coherence_score: {coherence_score}')
            for i, topic in topics:
                print(f"Topic {i}: {topic}")

        topic_counts = [0] * num_of_topics
        for bow in corpus:
            topic_dist = lda.get_document_topics(bow)
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] += 1
        if log_results:
            for i, count in enumerate(topic_counts):
                print(f"Topic {i}: {count} posts")

        return topics, coherence_score, lda, topic_counts

    def _preprocess_text(self, text):
        '''
        Copy paste from https://www.geeksforgeeks.org/nlp/topic-modeling-using-latent-dirichlet-allocation-lda/
        '''
        text = re.sub('\s+', ' ', text)  # Remove extra spaces
        text = re.sub('\S*@\S*\s?', '', text)  # Remove emails
        text = re.sub('\'', '', text)  # Remove apostrophes
        text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
        text = text.lower()  # Convert to lowercase
        return text

    def exercise2(self, log_results=True):
        '''
        Exercise 4.2 Sentiment: 
        Perform sentiment analysis on posts and comments. 
        What is the overall tone of the platform? How does sentiment vary across user posts discussing different topics identified in Exercise 3? 
        Please use VADER (nltk.sentiment) for this analysis. Answer and explain your queries/calculations below. 
        You may use SQL and/or Python to perform this task. (5 points)        
        '''
        sid = SentimentIntensityAnalyzer()
        for table in ['posts', 'comments']:
            if table not in self.tables:
                raise Exception(f"Table {table} must be inside the database to run exercise4")
            query = f"SELECT content FROM {table}"
            df = pd.read_sql_query(query, self.conn)
            if log_results:
                print(f"Rows in {table}: {len(df)}")
            # for content in df['content']:
            #     score = sid.polarity_scores(content)
            #     if log_results:
            #         print(f"Sentiment scores for content in {table}: {score}")

        # What is the overall tone of the platform?

        # How does sentiment vary across user posts discussing different topics identified in Exercise 3?

    def exercise3(self, log_results=True):
        '''
        Exercise 4.3 Learning from others’ mistakes:
        Find two social platforms similar to Mini Social that have been under fire for an engineering,
          design or operation error that severely affected a large group of users. 
        Describe how we can learn from their mistakes and draft up a plan about how Mini Social can be improved learning from their mistakes.
        You do not need to write code in this exercise unless your plan includes a specific change to an algorithm or function. (5 points)        
        
        Discord has changed database architecture multiple times first from Mongo to Cassandra and then to ScyllaDB. These database changes were needed because of rapid platform growth. 
        Mini Social is currently small enough to be handled by a single SQL Lite database, but in the future it might be necessary to upgrade to something more scalable.
        It's also important to know and test the database architecture throroughly. After Discord's migration to Cassandra they had an big issue with how the messages were deleted. 
        Instead of directly deleting the messages Cassandra creates thombstones to effectively skip deleted records. This caused the JVM to overload and trigger the "stop-the-world" error. 
        For users this was seen as slowness and eventually full downtime. If the Discord developers had know they would have simply lowered the tombstone lifespan to 2 days avoiding the overload.

        When the time comes Mini Social should consider other database options like PostgreSQL, Cassandra or ScyllaDB. Transformations should be done well ahead and tested throroughly.
        With Blackbox testing the new database can be tested alongside the old one to compare query results and performance.

        Slack had an Major outage in 2025 due to database shards which caused API breakdowns. As these sharding errors grow over time it's important to monitor API performance and error rates. 
        Currently Mini Social does not have any monitoring or alerting in place. Any slowly creeping performance issues are only detected when it's too late.
        For improvement each API endpoint should be montored for latency and errors. There should also be testing and alerting that triggers when the service starts failing.
        Long downtime causes the users to be unhappy and less confident with the platform.
        
        Disclaimer: The Slack source is and 3rd party company, which offers API intelligence services, but it still had the best reporting on the issue.
        
        https://discord.com/blog/how-discord-stores-billions-of-messages
        https://discord.com/blog/how-discord-stores-trillions-of-messages

        https://treblle.com/blog/slack-outage-api-failures
        https://www.tomsguide.com/news/live/slack-down-updates-outage-2-25

        '''

    def exercise4(self, log_results=True):
        # See app.py
        pass


if __name__ == "__main__":
    db_class = Week4(db_file='database.sqlite')
    #db_class.main()
    db_class.exercise1(log_results=True)
    # db_class.exercise2(log_results=False)
    # db_class.exercise3(log_results=False)
    # db_class.exercise4(log_results=True)