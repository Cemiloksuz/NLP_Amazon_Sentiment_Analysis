##############NLP_Amazon_Sentiment_Analysis############

# This is a sentiment analysis project.
# It includes the analysis of the comments received on the products of the Kozmos brand.
# The comments will be labeled by performing sentiment analysis and
# a classification model will be created with the labeled data.

#### Importing Library and Settings  ######

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('vader_lexicon')
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')

filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


###### Loading  The Date  #####

df = pd.read_excel("/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_14/amazon.xlsx")
df.head()
df.shape


# Normalizing Case Folding
###############################
df['Review'] = df['Review'].str.lower()

###############################
# Punctuations
###############################
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

###############################
# Numbers
###############################
df['Review'] = df['Review'].str.replace('\d', '')

###############################
# Stopwords
###############################
# nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords / Custom Words
###############################

sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

###############################
# Lemmatization
###############################

# nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['Review'].head(10)


############### Text Visualization ###############

""" Bar PLot"""
tf = df["Review"].apply(lambda x: pd.value_counts(x.split())).sum(axis = 0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending = False).head(10)

tf[tf["tf"] > 500].plot.bar(x = "words", y = "tf")
plt.show()

""" Word Cloud"""

all_text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size = 50,
                      max_words = 100,
                      background_color = "white").generate(all_text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

############# Sentiment Analysis #############

sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])


######### Sentiment Analysis  Feature Engineering #####


X_train, X_test, y_train, y_test = train_test_split(df["Review"],
                                                    df["sentiment_label"], random_state = 17)

tf_idf_word_vectorizer = TfidfVectorizer().fit(X_train)
X_train_tf_idf_word = tf_idf_word_vectorizer.transform(X_train)
X_test_tf_idf_word = tf_idf_word_vectorizer.transform(X_test)


######### Sentiment Analysis  Modelling ######

""" LOGISTIC REGRESSION """

log_model = LogisticRegression(random_state = 17).fit(X_train_tf_idf_word, y_train)

y_pred = log_model.predict(X_test_tf_idf_word)

print(classification_report(y_pred, y_test))

cross_val_score(log_model, X_test_tf_idf_word, y_test, cv = 5).mean()

# 0.8545983731570921

random_review = pd.Series(df["Review"].sample(1).values)
new_command = TfidfVectorizer().fit(X_train).transform(random_review)
pred = log_model.predict(new_command)
print(f"Random Review: {random_review[0]} \n Prediction: {pred} ")


""" RANDOM FOREST """

rf_model = RandomForestClassifier(random_state = 17)
rf_model.fit(X_train_tf_idf_word, y_train)
cross_val_score(rf_model, X_train_tf_idf_word, y_train, cv = 5).mean()

#  0.9120733432939522

# Hyperparameter Optimisation for Random Forest

rf_model = RandomForestClassifier(random_state = 17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv = 5,
                            n_jobs = -1,
                            verbose = 1).fit(X_train_tf_idf_word, y_train)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 17).fit(X_train_tf_idf_word, y_train)

cross_val_score(rf_final, X_train_tf_idf_word, y_train, cv = 5, n_jobs = -1).mean()

# 0.913023744496005

# The random forest model has more accuracy than the logistic regression model.






