#!/usr/bin/env python
# coding: utf-8

# # Text Mining Assignment

# In[1]:


import pandas as pd
import numpy as np 
import string 
import spacy 
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings('ignore')


# In[2]:


musk = pd.read_csv(r'C:\Users\gupta\Downloads\Data Science Assesment\Text Minning\Elon_musk.csv',encoding='latin',error_bad_lines = False)
musk


# In[3]:


musk=musk['Text']
musk


# In[4]:


# remove both the leading and the trailing characters
musk = [y.strip() for y in musk]
musk


# In[5]:


# removes empty strings, because they are considered in Python as False
musk = [x for x in musk if x] 
musk


# In[6]:


# Joining the list into one string/text
text = ' '.join(musk)
text


# In[7]:


#Punctuation
no_punc_text = text.translate(str.maketrans("\x92", "'", string.punctuation)) 
no_punc_text


# In[8]:


#Tokenization
from nltk.tokenize import word_tokenize

text_tokens = word_tokenize(no_punc_text)
text_tokens


# In[9]:


text_tokens[0:72]


# In[10]:


len(text_tokens)


# In[11]:


from nltk.corpus import stopwords


# In[12]:


my_stop_words = stopwords.words('english')
my_stop_words


# In[13]:


no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:65])


# In[14]:


#Noramalize the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:45])


# In[15]:


#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40])


# In[16]:


# NLP english language model of spacy library
nlp = spacy.load('en_core_web_sm')


# In[17]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])


# In[18]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:22])


# ### Feature Extraction :

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)


# In[20]:


pd.DataFrame.from_records([vectorizer.vocabulary_])


# In[21]:


pd.DataFrame.from_records([vectorizer.vocabulary_]).T


# In[22]:


pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(30)


# In[23]:


pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True).head(25)


# In[24]:


print(vectorizer.vocabulary_)


# In[25]:


print(vectorizer.get_feature_names()[50:100])


# In[26]:


print(X.toarray()[50:100])


# In[27]:


print(X.toarray().shape)


# #### Let's see how can bigrams and trigrams can be included here

# In[28]:


vectorizer_ngram_range = CountVectorizer(analyzer='word',
                                         ngram_range=(1,4),
                                         max_features = 100)

bow_matrix_ngram = vectorizer_ngram_range.fit_transform(musk)
bow_matrix_ngram


# In[29]:


print(vectorizer_ngram_range.get_feature_names())


# In[30]:


print(bow_matrix_ngram.toarray())


# #### TFidf vectorizer :

# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_n_gram_max_features = TfidfVectorizer(norm="",
                                                 analyzer='word',
                                                 ngram_range=(1,4),
                                                 max_features = 100)

tf_idf_matrix_n_gram_max_features = vectorizer_n_gram_max_features.fit_transform(musk)


# In[32]:


print(vectorizer_n_gram_max_features.get_feature_names())


# In[33]:


print(tf_idf_matrix_n_gram_max_features.toarray())


# ####  Generate wordcloud :

# In[34]:


# Define a function to plot word cloud

def plot_cloud(wordcloud):
    
    
    plt.figure(figsize=(60, 40))    # Set figure size

    
    plt.imshow(wordcloud)           # Display image
    
    
    plt.axis("off")                 # No axis details


# In[35]:


# Generate wordcloud
wordcloud = WordCloud(width = 3500, 
                      height = 2500,
                      background_color='black',
                      max_words=120,
                      colormap='Set2').generate(text)
# Plot
plot_cloud(wordcloud)


# In[36]:


musk2 = pd.read_csv('Elon_musk.csv', encoding='latin', error_bad_lines = False)
musk2


# In[37]:


musk2 = musk2['Text']
musk2


# In[38]:


musk2 = [x.strip() for x in musk2] # remove both the leading and the trailing characters
musk2 = [x for x in musk2 if x] # removes empty strings, because they are considered in Python as False
musk2[0:10]


# In[39]:


one_block = musk2[1]
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[40]:


one_block


# In[41]:


for token in doc_block[:]:
    print(token, token.pos_)


# In[42]:


#Filtering for nouns and verbs only
nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]

#print(nouns_verbs[5:25])
nouns_verbs


# In[43]:


len(nouns_verbs)


# In[44]:


#Counting tokens again
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
X


# In[45]:


X.shape


# In[46]:


X.toarray()


# In[47]:


cv.get_feature_names()


# In[48]:


#pd.DataFrame(X.toarray(), columns = sorted(cv.vocabulary_))
# Or
temp_df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
temp_df


# In[49]:


sum_words = X.sum(axis=0) #Column total
sum_words


# In[50]:


type(sum_words)


# In[51]:


cv.vocabulary_


# In[52]:


sorted(cv.vocabulary_)


# In[53]:


wf_df = pd.DataFrame({'word' : cv.get_feature_names(),
                      'count' : np.array(sum_words).flatten()})
wf_df


# #### Visualizing results :

# In[54]:


#Barchart for top 10 nouns + verbs
wf_df[0:15].plot.bar(x='word', figsize=(10,7), title='Top verbs and nouns')


# #### Emotion Mining :

# In[55]:


#Sentiment analysis
rohu = pd.read_csv('Afinn.csv', 
                   error_bad_lines=False, 
                   sep = ",", 
                   encoding = 'latin-1')
rohu


# In[56]:


rohu[1500:]


# In[57]:


from nltk import tokenize

sentences = tokenize.sent_tokenize(" ".join(musk2))

#sentences[5:15]
sentences


# In[58]:


sent_df = pd.DataFrame(sentences, columns=['sentence'])
sent_df


# In[59]:


affinity_scores = rohu.set_index('word')['value'].to_dict()
affinity_scores


# In[60]:


nlp = spacy.load('en_core_web_sm')


# In[61]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        #print(sentence)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0) #return 0 if key not found
    return sent_score


# In[62]:


# test that it works
calculate_sentiment(text = 'Amazing, wonderful session')


# In[63]:


calculate_sentiment(text = 'wonderful session')


# In[64]:


calculate_sentiment(text = 'great job, best explanation ever')


# In[65]:


calculate_sentiment(text = 'good')


# In[66]:


calculate_sentiment(text = 'What a ridiculous')


# In[67]:


calculate_sentiment(text = 'worst product and worst service ever')


# In[68]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[69]:


sent_df


# In[70]:


sent_df.iloc[4,0]


# In[71]:


# how many words are in the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
#sent_df['word_count'].head(10)
sent_df


# In[72]:


sent_df.sort_values(by='sentiment_value').tail(10)


# In[73]:


sent_df.sort_values(by='sentiment_value').head(15)


# In[74]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[75]:


# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0].head()


# In[76]:


sent_df[sent_df['sentiment_value']<-5]


# In[77]:


sent_df[sent_df['sentiment_value']<-5].head()['sentence']


# In[78]:


sent_df[sent_df['sentiment_value']<-5].head()['sentence'].tolist()


# In[79]:


sent_df[sent_df['sentiment_value']<-5].head()['sentence'].tolist()[0]


# In[80]:


sent_df['index'] = range(0, len(sent_df))


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])


# In[82]:


plt.figure(figsize=(14, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[83]:


sent_df.plot.scatter(x='word_count',
                     y='sentiment_value',
                     figsize=(9,9),
                     title='Sentence sentiment value to sentence word count')


# In[84]:


# PW for positive words 
# NW for Negative Words 


# In[86]:


PW=pd.read_csv('positive-words.txt', 
               error_bad_lines = False)
PW


# In[87]:


NW=pd.read_csv('negative-words.txt', 
               error_bad_lines = False,
               encoding='latin-1')
NW


# In[88]:


PW =PW.iloc[25:]
PW =PW.rename(columns={';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;':'Words'})
PW.reset_index(inplace=True)
PW =PW.drop('index',axis=1)
PW['Score'] = 1
PW


# In[89]:


NW = NW.iloc[25:]
NW = NW.rename(columns={';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;':'Words'})
NW.reset_index(inplace=True)
NW = NW.drop('index',axis=1)
NW['Score'] = -1
NW


# In[90]:


affinity_scores1 = PW.set_index('Words')['Score'].to_dict()
affinity_scores2 = NW.set_index('Words')['Score'].to_dict()
affinity_scores1.update(affinity_scores2)
affinity_scores = affinity_scores1
affinity_scores


# In[91]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        #print(sentence)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0) #return 0 if key not found
    return sent_score


# In[92]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[93]:


sent_df


# In[94]:


# how many words are in the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df


# In[95]:


sent_df.sort_values(by='sentiment_value').tail()


# In[96]:


sent_df.sort_values(by='sentiment_value').head()


# In[97]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[98]:


# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<0].head()


# In[99]:


sent_df[sent_df['sentiment_value']<-1]


# In[100]:


sent_df[sent_df['sentiment_value']<-1].head(20)['sentence']


# In[101]:


sent_df[sent_df['sentiment_value']<-1].head(5)['sentence'].tolist()


# In[102]:


sent_df['index'] = range(0, len(sent_df))


# In[103]:


sns.distplot(sent_df['sentiment_value'])


# In[104]:


plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[105]:


sent_df.plot.scatter(x='word_count',
                     y='sentiment_value',
                     figsize=(10,10),
                     title='Sentence sentiment value to sentence word count')


# ##### Positive and Negative words can't give us better results for sentimental analysis as compared to rohu(afinn dataset) which gives us better analysis.
