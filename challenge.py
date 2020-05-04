import json
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine


import sys
!{sys.executable} -m pip install psycopg2-binary


file_dir = '/Users/evafuentes/Desktop/Analysis_Projects/Movies_ETL/'


#Load the JSON into a List of Dictionaries
with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)
    
    
    
kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory = False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')


#INSPECT
wiki_movies_df = pd.DataFrame(wiki_movies_raw)


#INSPECT
wiki_movies_df.columns.tolist()


#use list comprehensions to filter data.
#Plan and Execute
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
len(wiki_movies)


#One thing to watch out for is to make nondestructive edits as
#much as possible while designing your pipeline. That means it’s
#better to keep your raw data in one variable, and put the cleaned
#data in another variable

wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


len(wiki_movies)


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


sorted(wiki_movies_df.columns.tolist())


#Step 1: Make an empty dict to hold all of the alternative titles.
#Step 2: Loop through a list of all alternative title keys.
#Step 2a: Check if the current key exists in the movie object.
#Step 2b: If so, remove the key-value pair and
#add to the alternative titles dictionary.
#Step 3: After looping through every key,
#add the alternative titles dict to the movie object.
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    return movie


clean_movies = [clean_movie(movie) for movie in wiki_movies]


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# rerun our list comprehension
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


#get the count of null values for each column
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


#tweak our list comprehension.
[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


box_office = wiki_movies_df['Box office'].dropna() 


def is_not_a_string(x):
    return type(x) != str


box_office[box_office.map(is_not_a_string)]

box_office[box_office.map(lambda x: type(x) != str)]


#use a simple space as our joining character and apply
#the join() function only when our data points are lists
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


import re


form_one = r'\$\d+\.?\d*\s*[mb]illion'


box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

#so that we dont get an error
box_office[~matches_form_one & ~matches_form_two]


form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'

form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+'

form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'

box_office.str.extract(f'({form_one}|{form_two})')

def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


#We no longer need the Box Office column, so we’ll just drop it:
wiki_movies_df.drop('Box office', axis=1, inplace=True)

budget = wiki_movies_df['Budget'].dropna()

budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]

budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]

wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

wiki_movies_df.drop('Budget', axis=1, inplace=True)

release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


#One way to parse those forms is with the following:
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'

#extract dates
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


wiki_movies_df.drop('Running time', axis=1, inplace=True)


kaggle_metadata.dtypes


kaggle_metadata['adult'].value_counts()


kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


kaggle_metadata['video'].value_counts()

kaggle_metadata['video'] == 'True'


kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


ratings.info(null_counts=True)


pd.to_datetime(ratings['timestamp'], unit='s')


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


movies_df[['title_wiki','title_kaggle']]


movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


#fill in the missing values with 0.
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


#make another scatter plot to compare the values
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


#The box_office and revenue columns are numeric, so we’ll make
#another scatter plot.
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


#Let’s look at the scatter plot for everything less than $1
#billion in box_office.
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')




#We’ll use the regular line plot (which can plot date data),
#and change the style to only put dots by adding style='.'
#to the plot() method:
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


#We should investigate that wild outlier around 2006.
#look for any movie whose release date according to
#Wikipedia is after 1996, but whose release date according
#to Kaggle is before 1965.
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


#it looks like somehow The Holiday in the Wikipedia
#data got merged with From Here to Eternity. We’ll
#have to drop that row from our DataFrame. We’ll get
#the index of that row with the following:
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index



#Then we can drop that row like this:
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


#see if there are any null values:
movies_df[movies_df['release_date_wiki'].isnull()]


#But the Kaggle data isn’t missing any release dates.
#In this case, we’ll just drop the Wikipedia data.

#For the language data, we’ll compare the value counts of each.
#movies_df['Language'].value_counts()
#We need to convert the lists in Language to tuples
#so that the value_counts() method will work

movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


#For the Kaggle data, there are no lists, so we can just run
#value_counts() on it.
movies_df['original_language'].value_counts(dropna=False)


#Production Companies
#we’ll start off just taking a look at a small number of samples.
movies_df[['Production company(s)','production_companies']]


#Putting it all together
#First, we’ll drop the title_wiki, release_date_wiki,
#Language, and Production company(s) columns.
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


#make a function that fills in missing data for a column pair
#and then drops the redundant column.

def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)
    
    
# run the function for the three column pairs that we decided to
#fill in zeros.
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


#it’s good to check that there aren’t any columns with
#only one value, since that doesn’t really provide any
#information. Don’t forget, we need to convert lists to
#tuples for value_counts() to work.
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


        
#Running this, we see that 'video' only has one value:
movies_df['video'].value_counts(dropna=False)


movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


#Finally, we need to rename the columns to be consistent.
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


#use a groupby on the “movieId” and “rating” columns
#and take the count for each group
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


#then we’ll rename the “userId” column to “count.”
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) 


#We can pivot this data so that movieId is the index,
#the columns will be all the rating values, and the
#rows will be the counts for each rating value.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')



#We want to rename the columns so they’re easier to
#understand. We’ll prepend rating_ to each column
#with a ****list comprehension****:
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


#we need to use a left merge, since we want to
#keep everything in movies_df:
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
movies_with_ratings_df


#because not every movie got a rating for each rating level,
#there will be missing values instead of zeros. We have to
#fill those in ourselves, like this:
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


from config import db_password


db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/new_movie_data"


engine = create_engine(db_string)


movies_df.to_sql(name='movies', con=engine)


#Step 2: Print Elapsed Time
import time
rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')




