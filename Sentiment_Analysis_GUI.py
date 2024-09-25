import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

#Specify the file path for Test Data Set
file_path = "Training Dataset/Original (No Pre-Processing)/Kaggle Sentiment Tweets (Original).csv"
test_data = pd.read_csv(file_path)

#Testing and Training Model on Test Dataset
normalised_test_data = test_data.copy()

x = normalised_test_data['text'].values
y = normalised_test_data["sentiment"].values

#Splitting The Data Into Training & Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

#Run The Model
model = SVC(C=1, kernel='linear')
model.fit(x_train_tfidf, y_train)

#Initialize session_state if it doesn't exist
if 'data' not in st.session_state:
    st.session_state.data = None

if 'normalised_data' not in st.session_state:
    st.session_state.normalised_data = None

#Load CSV function
def load_csv(uploaded_file):
    data = pd.read_csv(uploaded_file)
    columns_to_keep = [
    'id', 'likes', 'quotes', 'replies', 'retweets', 'searchQuery', 'text', 'timestamp', 'user/location', 'verified'
    ]

    data = data[columns_to_keep]
    return data

# Timestamp Convert
def check_and_convert_timestamp():
    if st.session_state.data is not None:
        try:
            st.session_state.data['timestamp'] = pd.to_datetime(st.session_state.data['timestamp'], format='%d/%m/%Y %H:%M')
        except ValueError:
            try:
                st.session_state.data['timestamp'] = pd.to_datetime(st.session_state.data['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                print("Error: Unable to parse timestamp in either format")

                # Check if 'timestamp' is present in data before accessing
                if 'timestamp' in st.session_state.data.columns:
                    st.session_state.data['month'] = st.session_state.data['timestamp'].dt.month
                    st.session_state.data['hours'] = st.session_state.data['timestamp'].dt.hour
                    st.session_state.data['day'] = st.session_state.data['timestamp'].dt.day


#Remove Outliers Function
def remove_outliers(data, variable, alpha=0.05):
    test_results = sm.OLS(data[variable], sm.add_constant(np.arange(len(data)))).fit().outlier_test()
    outliers = test_results[test_results['bonf(p)'] < alpha].index
    return data[~data.index.isin(outliers)]

#GUI PAGE ELEMENTS
menu_option = st.sidebar.selectbox(
    "Select a Page", ("Run Sentiment Analysis Model", "Sentiment Distribution", "Sentiment by Location", "Sentiment by Tweet Statistic", "Key Word Analysis"))



#RUN SENTIMENT ANALYSIS MODEL PAGE
if menu_option == "Run Sentiment Analysis Model":
    st.title("Run Sentiment Analysis Model")

    uploaded_file = st.file_uploader("Please Upload Your CSV File", type=["csv"])

    #Load data only if it's not already loaded
    if uploaded_file is not None and st.session_state.data is None:
        st.session_state.data = load_csv(uploaded_file)

    #Display the contents of the CSV file
    if st.session_state.data is not None:
        st.write("Dataframe before Model Run")
        st.write(st.session_state.data)

        #Button to run sentiment analysis and print the new DataFrame
        if st.button("Run Sentiment Analysis Model on Dataframe"):
            normalised_data = st.session_state.data.copy()
            tweets = normalised_data['text']
            tweets_tfidf = tfidf_vectorizer.transform(tweets)
            sentiment_prediction = model.predict(tweets_tfidf)
            normalised_data['sentiment'] = sentiment_prediction
            st.session_state.normalised_data = normalised_data
            st.write("DataFrame with Sentiment Analysis:")
            st.write(normalised_data[['id', 'text', 'sentiment']])



#SENTIMENT DISTRIBUTION PAGE
elif menu_option == "Sentiment Distribution":
    st.title("Sentiment Distributions")

    # Check if sentiment analysis has been performed
    if st.session_state.normalised_data is not None and 'sentiment' in st.session_state.normalised_data.columns:
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='sentiment', data=st.session_state.normalised_data, palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=ax)
        ax.set_title('Sentiment Distribution Bar Chart')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        sentiment_counts = st.session_state.normalised_data['sentiment'].value_counts()
        labels = sentiment_counts.index
        sizes = sentiment_counts.values

        # Define sentiment_palette
        sentiment_palette = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

        # Plot the pie chart with custom colors
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[sentiment_palette[label.lower()] for label in labels])
        ax.set_title('Sentiment Distribution Pie Chart')
        st.pyplot(fig)



#KEY WORD ANALYSIS PAGE
elif menu_option == "Key Word Analysis":
    if st.session_state.normalised_data is not None and 'sentiment' in st.session_state.normalised_data.columns:
      st.title("Most Used Keywords")



#WORDCLOUDS OF SENTIMENTS
      def generate_wordcloud(ax, sentiment):
          subset = st.session_state.normalised_data[st.session_state.normalised_data['sentiment'] == sentiment]
          text = ' '.join(subset['text'])

          stop_words = set(stopwords.words('english'))
          stop_words.update(['amazon', '&', 'amazon.', '.', '-', '@amazon', '3', 'via', ':', '#amazon', 'com', 'amzn', 'get', "it's"])

          wordcloud = WordCloud(width=400, height=200, background_color='white', stopwords=stop_words).generate(text)

          ax.imshow(wordcloud, interpolation='bilinear')
          ax.set_title(sentiment)
          ax.axis('off')

      sentiment_order = ['positive', 'negative', 'neutral']

      fig, axs = plt.subplots(1, len(sentiment_order), figsize=(12, 6))

      for i, sentiment in enumerate(sentiment_order):
          generate_wordcloud(axs[i], sentiment)

      st.pyplot(fig)



#POSITIVE KEYWORDS BAR CHART
      st.title("Top Positive Keywords")

      positive_df = st.session_state.normalised_data[st.session_state.normalised_data['sentiment'] == 'positive']
      positive_text = ' '.join(positive_df['text'])

      stop_words = set(stopwords.words('english'))
      stop_words.update(['amazon', '&', 'amazon.', '.', '-', '@amazon', '3', 'via', ':', '#amazon', 'com', 'amzn', 'get', "it's"])
      positive_words_list = [word.lower() for word in positive_text.split() if word.lower() not in stop_words]
      positive_word_counts = Counter(positive_words_list)

      df_positive_word_counts = pd.DataFrame(list(positive_word_counts.items()), columns=['Word', 'Count'])

      df_positive_word_counts = df_positive_word_counts.sort_values(by='Count', ascending=False)

      fig, ax = plt.subplots(figsize=(12, 6))
      sns.barplot(x='Count', y='Word', data=df_positive_word_counts.head(10), palette='Greens_r')
      ax.set(xlabel='Count', ylabel='Keyword', title='Top Positive Keywords')

      st.pyplot(fig)



#NEGATIVE KEYWORDS BAR CHART
      st.title("Top Negative Keywords")

      negative_df = st.session_state.normalised_data[st.session_state.normalised_data['sentiment'] == 'negative']
      negative_text = ' '.join(negative_df['text'])

      stop_words = set(stopwords.words('english'))
      stop_words.update(['amazon', '&', 'amazon.', '.', '-', '@amazon', '3', 'via', ':', '#amazon', 'com', 'amzn', 'get', "it's"])
      negative_words_list = [word.lower() for word in negative_text.split() if word.lower() not in stop_words]
      negative_word_counts = Counter(negative_words_list)

      df_negative_word_counts = pd.DataFrame(list(negative_word_counts.items()), columns=['Word', 'Count'])

      df_negative_word_counts = df_negative_word_counts.sort_values(by='Count', ascending=False)

      fig, ax = plt.subplots(figsize=(12, 6))
      sns.barplot(x='Count', y='Word', data=df_negative_word_counts.head(10), palette='Reds_r')
      ax.set(xlabel='Count', ylabel='Keyword', title='Top Negative Keywords')

      st.pyplot(fig)



    else:
      st.warning("Perform sentiment analysis first.")



#LOCATION BASED SENTIMENT
elif menu_option == "Sentiment by Location":
    if st.session_state.normalised_data is not None and 'sentiment' in st.session_state.normalised_data.columns:

#TOP 10 POSITIVE LOCATION SENTIMENT BAR CHART
      st.title("Top 10 User Locations with Positive Sentiments")

      filtered_positive_data = st.session_state.normalised_data[st.session_state.normalised_data['sentiment'] == 'positive'].dropna(subset=['sentiment', 'user/location'])

      top_positive_locations = filtered_positive_data['user/location'].value_counts().nlargest(10).index

      filtered_positive_data = filtered_positive_data[filtered_positive_data['user/location'].isin(top_positive_locations)]

      sns.set(style="whitegrid")

      fig, ax = plt.subplots(figsize=(16, 8))

      sns.countplot(x='user/location', data=filtered_positive_data, palette='Greens_r', ax=ax)
      plt.title('Top 10 User Locations with Positive Sentiments')
      plt.xlabel('User Location')
      plt.ylabel('Count')
      plt.xticks(rotation=45, ha='right')

      st.pyplot(fig)


#TOP 10 NEGATIVE LOCATION SENTIMENT BAR CHART
      st.title("Top 10 User Locations with Negative Sentiments")

      filtered_negative_data = st.session_state.normalised_data[st.session_state.normalised_data['sentiment'] == 'negative'].dropna(subset=['sentiment', 'user/location'])

      top_negative_locations = filtered_negative_data['user/location'].value_counts().nlargest(10).index

      filtered_negative_data = filtered_negative_data[filtered_negative_data['user/location'].isin(top_negative_locations)]

      sns.set(style="whitegrid")

      fig, ax = plt.subplots(figsize=(16, 8))

      sns.countplot(x='user/location', data=filtered_negative_data, palette='Reds_r', ax=ax)
      plt.title('Top 10 User Locations with Negative Sentiments')
      plt.xlabel('User Location')
      plt.ylabel('Count')
      plt.xticks(rotation=45, ha='right')

      st.pyplot(fig)


    else:
      st.warning("Perform sentiment analysis first.")


#TWEET STATISTIC PAGE
elif menu_option == "Sentiment by Tweet Statistic":
    if st.session_state.normalised_data is not None and 'sentiment' in st.session_state.normalised_data.columns:

      st.title('Sentiment by Tweet Length')

      st.session_state.normalised_data['Text_Length'] = st.session_state.normalised_data['text'].apply(len)

      sns.set(style="whitegrid")

      filtered_data_text_length = remove_outliers(st.session_state.normalised_data, 'Text_Length')

      fig, ax = plt.subplots(figsize=(10, 6))

      sns.boxplot(x='sentiment', y='Text_Length', data=filtered_data_text_length, palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=ax)
      plt.title('Box Plot for Text Length by Sentiment')
      plt.xlabel('Sentiment')
      plt.ylabel('Text Length')

      st.pyplot(fig)



      #SENTMENTS BY VERFIED STATUS
      st.title('Sentiments by Verified Status')

      sentiment_counts = st.session_state.normalised_data.groupby(['verified', 'sentiment']).size().reset_index(name='count')

      pivot_df = sentiment_counts.pivot(index='verified', columns='sentiment', values='count').fillna(0)

      colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

      fig, ax = plt.subplots()

      pivot_df.plot(kind='bar', stacked=True, color=[colors[col] for col in pivot_df.columns], ax=ax)

      for p in ax.patches:
          width, height = p.get_width(), p.get_height()
          x, y = p.get_xy()
          ax.annotate(f'{int(height)}', (x + width / 2, y + height / 2), ha='center', va='center', color='white', fontsize=10)

      plt.title('Sentiments by Verified Status')
      plt.xlabel('Verified Status')
      plt.ylabel('Count')
      plt.legend(title='Sentiment', bbox_to_anchor=(1, 1))

      st.pyplot(fig)



      #SENTIMENTS BASED ON LIKES VIOLIN PLOT
      st.title('Sentiments Based on Likes')

      fig, ax = plt.subplots(figsize=(12, 6))

      filtered_data_likes = remove_outliers(st.session_state.normalised_data, 'likes')

      sns.violinplot(x='sentiment', y='likes', data=filtered_data_likes, palette=['green', 'red', 'blue'], ax=ax)

      plt.title('Sentiments Based on Likes')
      plt.xlabel('Sentiment')
      plt.ylabel('Likes')

      st.pyplot(fig)



      #SENTIMENT BASED ON TWEET STATISTICS SCATTER PLOTS
      st.title('Sentiment on Specific Tweet Statistics')

      fig, axes = plt.subplots(2, 2, figsize=(16, 12))

      filtered_data_replies = remove_outliers(st.session_state.normalised_data, 'replies')
      sns.scatterplot(x='replies', y='sentiment', data=filtered_data_replies, hue='sentiment', palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=axes[0, 0])
      axes[0, 0].set_xlabel('Number of Replies')
      axes[0, 0].set_ylabel('Sentiment')
      axes[0, 0].legend().set_visible(False)

      filtered_data_quotes = remove_outliers(st.session_state.normalised_data, 'quotes')
      sns.scatterplot(x='quotes', y='sentiment', data=filtered_data_quotes, hue='sentiment', palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=axes[0, 1])
      axes[0, 1].set_xlabel('Number of Quotes')
      axes[0, 1].set_ylabel('Sentiment')
      axes[0, 1].legend().set_visible(False)

      filtered_data_likes = remove_outliers(st.session_state.normalised_data, 'likes')
      sns.scatterplot(x='likes', y='sentiment', data=filtered_data_likes, hue='sentiment', palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=axes[1, 0])
      axes[1, 0].set_xlabel('Number of Likes')
      axes[1, 0].set_ylabel('Sentiment')
      axes[1, 0].legend().set_visible(False)

      filtered_data_retweets = remove_outliers(st.session_state.normalised_data, 'retweets')
      sns.scatterplot(x='retweets', y='sentiment', data=filtered_data_retweets, hue='sentiment', palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, ax=axes[1, 1])
      axes[1, 1].set_xlabel('Number of Retweets')
      axes[1, 1].set_ylabel('Sentiment')
      axes[1, 1].legend().set_visible(False)

      plt.tight_layout()

      st.pyplot(fig)



    else:
      st.warning("Perform sentiment analysis first.")
