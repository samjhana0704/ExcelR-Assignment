#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
import matplotlib.pyplot as plt
import seaborn as sns


# ## __1 - Business Problem__  
# ___Recommend a best book based on the ratings___  

# ## __2 - Data collection and description__ 

# In[2]:


df = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Recmmendation system\book.csv, encoding = "ISO-8859-1")


# In[3]:


df1 = df.iloc[:,1:]


# In[4]:


df1.columns = ['userID', 'title', 'bookRating']


# In[5]:


df1.head()


# In[60]:


print('# of records: %d\n# of books: %d\n# of users: %d' % (len(df1), len(df1['title'].unique()), len(df1['userID'].unique())))


# In[94]:


palette = sns.color_palette("RdBu", 10)


# In[95]:


fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='bookRating', data=df1, palette=palette)
ax.set_title('Distribution of book ratings')

plt.show()


# ### The majority of ratings is between 5 and 10. Most often users tend to rate books for 8. Second the most frequent score is 7.

# ## __3 - Introduction to "Surprice" Package__ 

# In[6]:


df1.bookRating.unique()


# In[7]:


reader = surprise.Reader(rating_scale=(1, 10))


# In[8]:


data = surprise.Dataset.load_from_df(df1[['userID', 'title', 'bookRating']], reader)


# In[34]:


trainset, testset = train_test_split(data, test_size=.20)


# ## __4 - Finding the best algorithm for our Recommendation System__ 

# In[54]:


benchmark = []


# In[58]:


for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), BaselineOnly(), CoClustering()]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)


# In[59]:


pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')


# ### SVD () algorithm gave us the best rmse, therefore, we will train and predict with SVD

# ## __5 - Building our Recommendation System using surprice__ 

# In[35]:


algo = SVD()


# In[36]:


algo.fit(trainset)


# In[37]:


predictions = algo.test(testset)


# In[41]:


accuracy.rmse(predictions, verbose=True)


# In[61]:


def recommendation(userID):
    user = df1.copy()
    already_read = df1[df1['userID'] == userID]['title'].unique()
    user = user.reset_index()
    user = user[~user['title'].isin(already_read)]
    user['Estimate_Score']=user['title'].apply(lambda x: algo.predict(userID, x).est)
    #user = user.drop('title', axis = 1)
    user = user.sort_values('Estimate_Score', ascending=False)
    print(user.head(10))


# ## __6 - Building our Recommendation System using correlation__ 

# In[64]:


rating = pd.pivot_table(df1, index='userID', values='bookRating', columns='title', fill_value=0)


# In[65]:


corr = np.corrcoef(each_book_rating.T)


# In[66]:


corr.shape


# In[67]:


book_list=  list(rating)


# In[68]:


book_titles =[] 


# In[69]:


for i in range(len(book_list)):
    book_titles.append(book_list[i])


# In[70]:


book_titles


# In[77]:


def get_recommendation(books_list):
    book_similarities = np.zeros(corr.shape[0])
    
    for book in books_list:
        book_index = book_titles.index(book)
        book_similarities += corr[book_index] 
        book_preferences = []
    for i in range(len(book_titles)):
        book_preferences.append((book_titles[i],book_similarities[i]))
        
    return sorted(book_preferences, key= lambda x: x[1], reverse=True)


# ## __7 - Conclusion__ 

# In[62]:


recommendation(276747)


# ### You can enter any User ID and get the recommendation and estimated score

# In[73]:


my_fav_books = ['Classical Mythology','Clara Callan']


# In[85]:


print('The books you might like: \n' , get_recommendation(my_fav_books)[:10])


# ### You can enter your favourite book and get the recommendation on what might like

# In[ ]:




