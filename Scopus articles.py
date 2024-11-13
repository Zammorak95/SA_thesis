import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Import and process CSV files

# Specify the path to the directory containing your CSV files
csv_directory = r'C:\Users\sheri\OneDrive - University of Twente\Thesis\Scopus Articles\Search query'

# Use os.listdir to get a list of all files in the specified directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each CSV file and append its data to the list
for file in csv_files:
    # Construct the full path to the CSV file
    file_path = os.path.join(csv_directory, file)

    # Read the CSV file into a DataFrame and append it to the list
    current_df = pd.read_csv(file_path)
    dfs.append(current_df)

# Concatenate the list of DataFrames into one DataFrame
articles_df = pd.concat(dfs, ignore_index=True)

articles_df.drop_duplicates(subset='Title', keep='first', inplace=True)

# Keep only necessary columns

columns_to_keep = [
    'Authors',
    'Title',
    'Abstract',
    'Year',
    'Cited by',
    'Source title',
    'Publisher',
    'Document Type',
    'DOI',
    'Author Keywords',
    'Index Keywords'
]

articles_df = articles_df[columns_to_keep]

#In Document Type keep only papers

articles_df = articles_df[articles_df['Document Type'].isin(['Article', 'Conference paper'])]

# Load the impact scores from the CSV file
impact_scores_df = pd.read_csv(r'C:\Users\sheri\OneDrive - University of Twente\Thesis\unique_sources.csv',sep=';', decimal=',')

# If the column names are different, please adjust accordingly
impact_scores_df = impact_scores_df[['Source title', 'Impact Score']]

# Merge the impact scores into articles_df based on 'Source title'
articles_df = pd.merge(articles_df, impact_scores_df, how='left', on='Source title')

# Drop rows with NaN values in the 'Impact Score' column
articles_df = articles_df.dropna(subset=['Impact Score'])

## Determine similarity using TfidVectorizer
# Keywords including variations
keywords = [
    "artificial intelligence", 'AI', 'ML', "Neural NetWork",
    "challenges", "limitations", 'Comparative Study', 'literature overview', "sentiment analysis", "machine learning",
    'Investor sentiment', "review", "models", "financial markets", 'NLP'
    "customer sentiment analysis", "stock markets", 'Time-series analysis',
    "natural language processing", "stock prices",
    'Literature Review', 'Systematic Review', 'Research Gap Analysis', 'news analysis', 'recurrent neural network',
    'RNN', 'news', "bibliometric review"
]
keywords = [keyword.lower() for keyword in keywords]

# Define weights for specific keywords (1.0 is the default weight)
keyword_weights = {
    'artificial intelligence': 2,
    'sentiment analysis': 1.8,
    'literature review': 1.6,
    'news analysis': 1.2,
    'stock markets': 1.2,
    'systematic review': 1.8
    # Add other keyword weights as needed
}

# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

# Combine 'Title' and 'Abstract' columns into a single column for analysis
articles_df['Title and Abstract'] = articles_df['Title'] + ' ' + articles_df['Abstract']
articles_df['Title and Abstract'] = articles_df['Title and Abstract'].str.lower()

# Fit and transform the text data using TfidfVectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(articles_df['Title and Abstract'])

# Transform the keywords using the same vectorizer
keywords_tfidf = tfidf_vectorizer.transform(keywords)

# Multiply TF-IDF values by weights for specific keywords
for keyword, weight in keyword_weights.items():
    idx = keywords.index(keyword)
    keywords_tfidf[:, idx] *= weight

# Determine similarity between articles and keywords
cosine_similarities = cosine_similarity(tfidf_matrix, keywords_tfidf)

# Add a new column 'Similarity' to the DataFrame
articles_df['Similarity'] = cosine_similarities.mean(axis=1)

# Sort the DataFrame by 'Similarity' column in descending order
articles_df = articles_df.sort_values(by='Similarity', ascending=False)

# Reset the index after sorting
articles_df.reset_index(drop=True, inplace=True)

# Define thresholds for Similarity and Impact Score
top_similarity_threshold = articles_df['Similarity'].mean() + (articles_df['Similarity'].std() * 2)
top_impact_score_threshold = articles_df['Impact Score'].mean() + (articles_df['Impact Score'].std() * 1)
similarity_threshold = articles_df['Similarity'].mean() + (articles_df['Similarity'].std() * 0.75)
impact_score_threshold = articles_df['Impact Score'].mean() + (articles_df['Impact Score'].std() * 0.5)

# Filter rows based on the thresholds
top_relevant_articles_df = articles_df[(articles_df['Similarity'] > top_similarity_threshold) &
                                   (articles_df['Impact Score'] > top_impact_score_threshold)]

top_relevant_articles_df = top_relevant_articles_df.sort_values(by='Similarity', ascending=False)

top_relevant_articles_df.reset_index(drop=True, inplace=True)

relevant_articles_df = articles_df[(articles_df['Similarity'] > similarity_threshold) &
                                   (articles_df['Impact Score'] > impact_score_threshold)]

relevant_articles_df = relevant_articles_df.sort_values(by='Similarity', ascending=False)

relevant_articles_df.reset_index(drop=True, inplace=True)



# print(relevant_articles_df['Title'])
print('Amount of r articles: ', relevant_articles_df.shape[0])

# print(top_relevant_articles_df['Title'])
print('Amount of t articles: ', top_relevant_articles_df.shape[0])

print('Impact score mean: ', articles_df['Impact Score'].mean())
print('Impact score std: ', articles_df['Impact Score'].std())
print('Similarity score mean: ', articles_df['Similarity'].mean())
print('Similarity score std: ', articles_df['Similarity'].std())
print(top_relevant_articles_df['Title'].head(10))
print(relevant_articles_df['Title'].head(10))

top_relevant_articles_df.to_csv('Top_relevant_articles.csv')
relevant_articles_df.to_csv('Relevant_articles.csv', index=True)
# print(relevant_articles_df.info())
print('End of Code')
