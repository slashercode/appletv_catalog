# ----------------------#
# IMPORT LIBRARIES
# ----------------------#
import warnings
from math import ceil

import kaleido
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
from ast import literal_eval

import missingno as msno
import plotly.express as px
from matplotlib import style

# load the titles and credits dataset
titles = pd.read_csv("titles.csv")
credits = pd.read_csv("credits.csv")

# merge the datasets into one file
merged_dataset = pd.merge(titles, credits, on="id")
merged_dataset.to_csv("merged_data.csv", index=False)

# constant values
TITLE_FONT = 25
LABEL_FONT = 15
TICK_FONT = 20

# ----------------------#
# INFORMATION ABOUT THE DATASETS
# ----------------------#


def shape_of_dataset(df, dataset_name="df"):
    """
    Gets the the number of rows and columns in the DataFrame
    Arguments:
            df: DataFrame for which we want to get the number of rows and columns
            dataset_name (optinal): name of the dataset
    Return:
            tuple containing two integers, which represent the number of rows and columns in the input DataFrame
    """

    print(
        f"{dataset_name} dataset has {df.shape[0]} nrows and {df.shape[1]} ncolumns \n"
    )
    return df.shape[0], df.shape[1]


# shape of the dataset
titles_r, titles_c = shape_of_dataset(titles, "Titles")
credits_r, credits_c = shape_of_dataset(credits, "Credits")


def count_null_values(df, dataset_name):
    """
    Counts the number of null values in the DataFrame
    Arguments:
            df: DataFrame for which we want to get the number of null values
            dataset_name: name of the dataset
    Return:
            number of null values in the input DataFrame
    """

    num_of_total_null_values = sum(df.isnull().sum().values)
    print(f"{dataset_name} dataset has {num_of_total_null_values} null values\n")
    return num_of_total_null_values


# number of null values of the dataset
titles_null = count_null_values(titles, "Titles")
credits_null = count_null_values(credits, "Credits")


def detect_null_columns(df, dataset_name):
    """
    Detects the number of null values in the columns in the DataFrame
    Arguments:
            df: DataFrame for which we want to get the number of null values of the columns
            dataset_name: name of the dataset
    Return:
            tuple that contains the list of column names with null values and the number of columns with null values
    """
    col = []
    s = df.isnull().sum()
    for x in range(len(s)):
        if s[x] > 0:
            col.append(s.index[x])
    tot_cols = len(col)
    if tot_cols == 0:
        print(f"{dataset_name} dataset has no null columns\n")
    else:
        print(f"{dataset_name} dataset has {tot_cols} null columns and they are:")
        for x in col:
            print(x, end=",")
        print()
    return col, len(col)


# number of null values in the columns of the dataset
total_titles_null_cols, titles_null_cols = detect_null_columns(titles, "Titles")
total_credits_null_cols, credits_null_cols = detect_null_columns(credits, "Credits")

detailed_db = pd.DataFrame(
    {
        "dataset": [],
        "nrows": [],
        "ncols": [],
        "null_amount": [],
        "names_null_cols": [],
        "num_null_cols": [],
    }
)


def fill_db_dataset(
    dataset_name, nrows, ncols, null_amount, name_null_cols, num_null_cols
):
    """
    Fills the database with complete information
    Arguments:
            dataset_name: name of the dataset
            nrows: number of rows in the dataset
            ncols: number of columns in the dataset
            null_amount: total number of null values in the dataset
            name_null_cols: names of columns that contain null values
            num_null_cols: number of columns that contain null values
    Return:
            None
    """
    detailed_db.loc[len(detailed_db.index)] = [
        dataset_name,
        nrows,
        ncols,
        null_amount,
        ", ".join(name_null_cols),
        int(num_null_cols),
    ]


# An Example
fill_db_dataset(
    "Olist Customer",
    titles_r,
    titles_c,
    titles_null,
    total_titles_null_cols,
    titles_null_cols,
)
fill_db_dataset(
    "Olist Geolocation",
    credits_r,
    credits_c,
    credits_null,
    total_credits_null_cols,
    credits_null_cols,
)
print(detailed_db)


titles.describe().T.style.set_properties(
    **{
        "background-color": "#FBA7A7",
        "font-size": "17px",
        "color": "#ffffff",
        "border-radius": "1px",
        "border": "1.5px solid black",
    }
)
credits.describe().T.style.set_properties(
    **{
        "background-color": "#FBA7A7",
        "font-size": "17px",
        "color": "#ffffff",
        "border-radius": "1px",
        "border": "1.5px solid black",
    }
)

# create a matrix plot to show the pattern of missing values in the dataset
msno.matrix(titles)
# set title
plt.title(
    "Distribution of Missing Values in titles dataset",
    fontsize=TITLE_FONT,
    fontstyle="oblique",
)
# save the plot
plt.savefig("fig1.png", bbox_inches="tight")

# create a matrix plot to show the pattern of missing values in the dataset
msno.matrix(credits)
# set title
plt.title(
    "Distribution of Missing Values in credits dataset",
    fontsize=TITLE_FONT,
    fontstyle="oblique",
)
# save the plot
plt.savefig("fig2.png", bbox_inches="tight")


# ----------------------#
# DATA ANALYSIS AND VISUALIZATION
# ----------------------#


def count_plot(
    x,
    df,
    title,
    xlabel,
    ylabel,
    width,
    height,
    order=None,
    rotation=False,
    palette="winter",
    hue=None,
    plot_num=1,
):
    """
    Creates a count plot based on a specified variable x in a given dataset
    Arguments:
            x: the name of the column in the dataframe that contains the values to be to plotled on the x-axis
            df: the DataFrame containing the data to be plotted
            title: the title of the plot
            xlabel: the label for the x-axis
            ylabel: the label for the y-axis
            width: the width of the plot
            height: the height of the plot
            order (optional): specifies the order of the categories in the plot
            rotation: a boolean parameter that specifies whether to rotate the x-axis labels
            palette: the color palette to use for the plot
            hue (optional): specifies the column in the dataframe to use for grouping the data
            plot_num: number of the plot figure to be saved
    Return:
            None
    """
    ncount = len(df)
    plt.figure(figsize=(width, height))
    ax = sns.countplot(x=x, palette=palette, order=order, hue=hue)
    plt.title(title, fontsize=TITLE_FONT)
    if rotation:
        plt.xticks(rotation="vertical", fontsize=TICK_FONT)
    plt.xlabel(xlabel, fontsize=LABEL_FONT)
    plt.ylabel(ylabel, fontsize=LABEL_FONT)

    ax.yaxis.set_label_position("left")
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(
            "{:.1f}%".format(100.0 * y / ncount),
            (x.mean(), y),
            ha="center",
            va="bottom",
        )  # set the alignment of the text

    plt.savefig(f"fig{plot_num}.png", bbox_inches="tight")


# countplot of the types of titles
x = titles["type"]
count_plot(
    x, titles, "Movies vs shows frequency", "Type", "Frequency", 12, 8, plot_num=3
)
## 63.5% OF THE RECORDS ARE TV-SHOWS ##

# countplot of the age of certification of titles
x = titles["age_certification"]
order = titles["age_certification"].value_counts().index
count_plot(
    x,
    titles,
    "age certification frequency",
    "age certification",
    "Frequency",
    12,
    8,
    order=order,
    palette="summer",
    plot_num=4,
)
## 30% OF THE MOVIES AND SHOWS ARE TV-MA WHICH IS SPECIFICALLY DESIGNED TO BE VIEWED BY ##
## ADULTS AND THEREFORE MAY BE UNSUITABLE FOR CHILDREN UNDER 17 ##


def bar_plot(
    x,
    y,
    df,
    annotation,
    title,
    xlabel,
    ylabel,
    width,
    height,
    order=None,
    rotation=False,
    palette="winter",
    hue=None,
    plot_num=1,
):
    """
    Creates a vertical bar plot based on a specified variable x in a given dataset
    Arguments:
            x: the name of the column in the dataframe that contains the values to be to plotled on the x-axis
            y: the name of the column in the dataframe that contains the values to be to plotled on the y-axis
            df: the DataFrame containing the data to be plotted
            annotation: list of values that will be annotated on top of each bar
            title: the title of the plot
            xlabel: the label for the x-axis
            ylabel: the label for the y-axis
            width: the width of the plot
            height: the height of the plot
            order (optional): specifies the order of the categories in the plot
            rotation: a boolean parameter that specifies whether to rotate the x-axis labels
            palette: the color palette to use for the plot
            hue (optional): specifies the column in the dataframe to use for grouping the data
            plot_num: number of the plot figure to be saved
    Return:
            None
    """
    ncount = len(df)
    plt.figure(figsize=(width, height))
    ax = sns.barplot(x=x, y=y, palette=palette, order=order, hue=hue)
    plt.title(title, fontsize=TITLE_FONT)
    if rotation:
        plt.xticks(rotation="vertical", fontsize=TICK_FONT)
    plt.xlabel(xlabel, fontsize=LABEL_FONT)
    plt.ylabel(ylabel, fontsize=LABEL_FONT)

    ax.yaxis.set_label_position("left")
    c = 0
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(
            "{:.1f}".format(annotation[c]),
            (x.mean(), y),
            ha="center",
            va="bottom",
            color="black",
        )  # set the alignment of the text
        c += 1
    plt.savefig(f"fig{plot_num}.png", bbox_inches="tight")


movies = titles[titles["type"] == "MOVIE"]
shows = titles[titles["type"] == "SHOW"]
# number of movies and shows in the dataset
print(f"Dataset contains {len(movies)} movie and {len(shows)} show")

# ----------------------#
# 1.A: TOP MOVIES ON IMDB (BASED ON SCORES)
# ----------------------#

# top 20 scored movies on IMDB
top20_imdb_scores_movies = (
    movies.groupby("title")["imdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
# barplot of the top 20 rated movies
x = top20_imdb_scores_movies["title"]
y = top20_imdb_scores_movies["imdb_score"]
bar_plot(
    x,
    y,
    top20_imdb_scores_movies,
    top20_imdb_scores_movies["imdb_score"],
    "Top 20 rated AppleTV movies on imdb",
    "Movie name",
    "Rating",
    12,
    8,
    rotation=True,
    palette="dark",
    plot_num=5,
)
## 'COME FROM AWAY' IS THE HIGHLY RATED APPLETV MOVIE ON IMDB WITH 8.6 RATINGS ##

merg = top20_imdb_scores_movies.merge(movies[["title", "imdb_votes"]], on="title")
merg.isnull().sum()
# only one missing value and it is the number of votes of 'Blush' movie
merg.fillna(
    1800.0, inplace=True
)  # 1800 is the number of votes of the 'Blush' movie on IMDB

plt.style.use("fivethirtyeight")
x = merg["title"]
y1 = merg["imdb_score"]
y2 = merg["imdb_votes"]
# create a new figure
fig, ax1 = plt.subplots()

# set the dimension of the figure
fig.set_figheight(10)
fig.set_figwidth(15)
# create a twin y-axis for the figure
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#AF7595")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title("Ratings and votes per movie", color="black", fontsize=TITLE_FONT)
ax1.set_ylabel("Ratings", color="#AF7595", fontsize=LABEL_FONT)
ax2.set_ylabel("Votings", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="green", fontsize=TICK_FONT)
c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:.1f}".format(merg["imdb_score"][c]),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
        fontsize=LABEL_FONT,
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig6.png", bbox_inches="tight")

## 'CODA' MOVIE AN INDICATION THAT IT IS A GOOD MOVIE, AS THE NUMBER OF VOTES EXCEEDED 140K AND ITS RATING IS 8.0 ##
## YOU CANNOT BE CERTAIN THAT THE 'COME FROM AWAY' MOVIE IS THE BEST BECAUSE IT IS THE HIGHEST RATED. THIS IS SIMPLY ##
## BECAUSE THE NUMBER OF VOTES DOES NOT EXCEED 2,500 ##

# ----------------------#
# 1.B: TOP MOVIES ON IMDB (BASED ON VOTES)
# ----------------------#
# top 20 voted movies on IMDB
top20_imdb_voting_movies = (
    movies.groupby("title")["imdb_votes"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)


def bar_plot2(
    x,
    y,
    df,
    annotation,
    title,
    xlabel,
    ylabel,
    width,
    height,
    order=None,
    rotation=False,
    palette="winter",
    hue=None,
    plot_num=1,
):
    """
    Creates a vertical bar plot based on a specified variable x in a given dataset
    Arguments:
            x: the name of the column in the dataframe that contains the values to be to plotled on the x-axis
            y: the name of the column in the dataframe that contains the values to be to plotled on the y-axis
            df: the DataFrame containing the data to be plotted
            annotation: list of values that will be annotated on top of each bar
            title: the title of the plot
            xlabel: the label for the x-axis
            ylabel: the label for the y-axis
            width: the width of the plot
            height: the height of the plot
            order (optional): specifies the order of the categories in the plot
            rotation: a boolean parameter that specifies whether to rotate the x-axis labels
            palette: the color palette to use for the plot
            hue (optional): specifies the column in the dataframe to use for grouping the data
            plot_num: number of the plot figure to be saved
    Return:
            None
    """
    ncount = len(df)
    plt.figure(figsize=(width, height))
    ax = sns.barplot(x=x, y=y, palette=palette, order=order, hue=hue)
    plt.title(title, fontsize=TITLE_FONT)
    if rotation:
        plt.xticks(rotation="vertical", fontsize=TICK_FONT)
    plt.xlabel(xlabel, fontsize=LABEL_FONT)
    plt.ylabel(ylabel, fontsize=LABEL_FONT)

    ax.yaxis.set_label_position("left")
    c = 0
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(
            "{:}".format(ceil(annotation[c])),
            (x.mean(), y),
            ha="center",
            va="bottom",
            color="black",
        )  # set the alignment of the text
        c += 1
    # save the plot
    plt.savefig(f"fig{plot_num}.png", bbox_inches="tight")


plt.style.use("ggplot")
x = top20_imdb_voting_movies["title"]
y = top20_imdb_voting_movies["imdb_votes"]
bar_plot2(
    x,
    y,
    top20_imdb_voting_movies,
    top20_imdb_voting_movies["imdb_votes"],
    "Top 20 most voted AppleTV movies on IMDB",
    "Movie name",
    "Rating",
    15,
    8,
    rotation=True,
    palette="flare",
    plot_num=7,
)
## 'THE ITALIAN JOB' GOT MORE THAN 370K VOTINGS ##
## WE MUST CARE ABOUT THE NUMBER OF VOTES BECAUSE THIS IS AN INDICATOR THAT THE MOVIE HAS BEEN ##
## WATCHED BY A LOT OF PEOPLE

merg = top20_imdb_voting_movies.merge(movies[["title", "imdb_score"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["imdb_votes"]
y2 = merg["imdb_score"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#32746D")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title("Ratings and votes per movie", color="black", fontsize=TITLE_FONT)
ax1.set_ylabel("Votings", color="blue", fontsize=LABEL_FONT)
ax2.set_ylabel("Ratings", color="#32746D", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:}".format(ceil(merg["imdb_votes"][c])),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig8.png", bbox_inches="tight")

## 'THE ITALIAN JOB' HAD MORE THAN 370K VOTES WITH 7 STARS RATINGS ##
## ACCORDING TO STATS(IMDB RATINGS AND VOTRES), IT IS SEEN THAT THE 'ITALIAN JOB', 'CODA', AND 'A CHARLIE ##
## BROWN CHRISTMAS' ARE THE MOST INTERESTING MOVIES


# ----------------------#
# 2.A: TOP MOVIES ON TMDB (BASED ON SCORES)
# ----------------------#
# top 20 scored movies on TMDB
top20_tmdb_scores_movies = (
    movies.groupby("title")["tmdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
x = top20_tmdb_scores_movies["title"]
y = top20_tmdb_scores_movies["tmdb_score"]
bar_plot(
    x,
    y,
    top20_tmdb_scores_movies,
    top20_tmdb_scores_movies["tmdb_score"],
    "Top 20 rated AppleTV movies on TMDB",
    "Movie name",
    "Ratings",
    12,
    8,
    rotation=True,
    palette="spring",
    plot_num=9,
)
## 'THE BOY, THE MOLE, THE FOX AND THE HORSE' IS THE HIGHLY RATED MOVIE ON TMDB WITH A 8.6 RATINGS ##

merg = top20_tmdb_scores_movies.merge(movies[["title", "tmdb_popularity"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["tmdb_score"]
y2 = merg["tmdb_popularity"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#D96C06")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title(
    "Top 20 Rated movies with Popularity on TMDB", color="black", fontsize=TITLE_FONT
)
ax1.set_ylabel("Ratings", color="#D96C06", fontsize=LABEL_FONT)
ax2.set_ylabel("Popularity", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:.1f}".format(merg["tmdb_score"][c]),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig10.png", bbox_inches="tight")
## 'EMANCIPATION' AND 'LUCK' MOVIES HAVE A DECENT RATING AND A VERY HIGH POPULARITY ##

# ----------------------#
# 2.B: TOP MOVIES ON TMDB (BASED ON POPULARITY)
# ----------------------#
# top 20 popular movies on TMDB
top20_tmdb_pop_movies = (
    movies.groupby("title")["tmdb_popularity"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
x = top20_tmdb_pop_movies["title"]
y = top20_tmdb_pop_movies["tmdb_popularity"]
bar_plot(
    x,
    y,
    top20_tmdb_pop_movies,
    top20_tmdb_pop_movies["tmdb_popularity"],
    "Top 20 most popular AppleTV movies on TMDB",
    "Movie name",
    "Popularity",
    12,
    8,
    rotation=True,
    palette="autumn",
    plot_num=11,
)
## LUCK IS THE MOST POPULAR MOVIE ON TMDB

merg = top20_tmdb_pop_movies.merge(movies[["title", "tmdb_score"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["tmdb_popularity"]
y2 = merg["tmdb_score"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#453750")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title(
    "Top 20 most popular movies with ratings on TMDB",
    color="black",
    fontsize=TITLE_FONT,
)
ax1.set_ylabel("Popularity", color="#453750", fontsize=LABEL_FONT)
ax2.set_ylabel("Ratings", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:.1f}".format(merg["tmdb_popularity"][c]),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
        fontsize=18,
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig12.png", bbox_inches="tight")


# ----------------------#
# 3.A: TOP SHOWS ON IMDB (SCORES)
# ----------------------#
# top 20 scored shows on IMDB
top20_imdb_scores_shows = (
    shows.groupby("title")["imdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
x = top20_imdb_scores_shows["title"]
y = top20_imdb_scores_shows["imdb_score"]
bar_plot(
    x,
    y,
    top20_imdb_scores_shows,
    top20_imdb_scores_shows["imdb_score"],
    "Top 20 rated AppleTV shows on IMDB",
    "Show name",
    "Rating",
    12,
    8,
    rotation=True,
    palette="flare",
    plot_num=13,
)
## SLUMBERKINS IS THE MOST RATED APPLETV SHOW ON IMDB WITH A 9.5 RATINGS ##

merg = top20_imdb_scores_shows.merge(shows[["title", "imdb_votes"]], on="title")
# no null values
merg.isnull().sum()
plt.style.use("fivethirtyeight")
x = merg["title"]
y1 = merg["imdb_score"]
y2 = merg["imdb_votes"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#AF7595")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title("Top 20 Rated shows with votes on IMDB", color="black", fontsize=TITLE_FONT)
ax1.set_ylabel("Ratings", color="#AF7595", fontsize=LABEL_FONT)
ax2.set_ylabel("Votings", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="green", fontsize=TICK_FONT)

c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:.1f}".format(merg["imdb_score"][c]),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
        fontsize=18,
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig14.png", bbox_inches="tight")
## TED LASSO HAS 8.8 RATINGS AND OVER 220K VOTES WHICH IS DEFINITELY A GOOD INDICATOR TO BE ##
## THE BEST SHOW ON THE LIST ##

# ----------------------#
# 3.B: TOP SHOWS ON IMDB (BASED ON VOTES)
# ----------------------#
# top 20 voted shows on IMDB
top20_imdb_voting_shows = (
    shows.groupby("title")["imdb_votes"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
plt.style.use("ggplot")
x = top20_imdb_voting_shows["title"]
y = top20_imdb_voting_shows["imdb_votes"]
bar_plot2(
    x,
    y,
    top20_imdb_voting_shows,
    top20_imdb_voting_shows["imdb_votes"],
    "Top 20 most voted AppleTV Shows on IMDB",
    "Show name",
    "Rating",
    15,
    8,
    rotation=True,
    palette="flare",
    plot_num=15,
)
merg = top20_imdb_voting_shows.merge(shows[["title", "imdb_score"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["imdb_votes"]
y2 = merg["imdb_score"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#32746D")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title(
    "Top 20 most voted shows with ratings on IMDB", color="black", fontsize=TITLE_FONT
)
ax1.set_ylabel("Votings", color="#32746D", fontsize=LABEL_FONT)
ax2.set_ylabel("Ratings", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:}".format(ceil(merg["imdb_votes"][c])),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig16.png", bbox_inches="tight")
## TED LASSO BY FAR (ACCORDING TO STATS: NUMBER OF VOTES AND RATINGS) IS THE BEST SHOW ##

# ----------------------#
# 4.A: TOP SHOWS ON TMDB (BASED ON SCORES)
# ----------------------#
# top 20 voted shows on TMDB
top20_tmdb_scores_shows = (
    shows.groupby("title")["tmdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
x = top20_tmdb_scores_shows["title"]
y = top20_tmdb_scores_shows["tmdb_score"]
bar_plot(
    x,
    y,
    top20_tmdb_scores_shows,
    top20_tmdb_scores_shows["tmdb_score"],
    "Top 20 rated AppleTV shows on TMDB",
    "Show name",
    "Ratings",
    12,
    8,
    rotation=True,
    palette="spring",
    plot_num=17,
)
## HELPSTERS HELP YOU IS THE HIGHEST RATED ON TMDB ##

merg = top20_tmdb_scores_shows.merge(shows[["title", "tmdb_popularity"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["tmdb_score"]
y2 = merg["tmdb_popularity"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#D96C06")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title(
    "Top 20 Rated shows with Popularity on TMDB", color="black", fontsize=TITLE_FONT
)
ax1.set_ylabel("Ratings", color="#D96C06", fontsize=LABEL_FONT)
ax2.set_ylabel("Popularity", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:.1f}".format(merg["tmdb_score"][c]),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig18.png", bbox_inches="tight")
## IT IS SEEN THAT ALL 10/10 RATED MOVIES HAVE VERY LOW POPULARITY ##

# ----------------------#
# 4.A: TOP SHOWS ON TMDB (BASED ON POPULARITY)
# ----------------------#
# top 20 popular shows on TMDB
top20_tmdb_pop_shows = (
    shows.groupby("title")["tmdb_popularity"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
    .head(20)
)
x = top20_tmdb_pop_shows["title"]
y = top20_tmdb_pop_shows["tmdb_popularity"]
bar_plot(
    x,
    y,
    top20_tmdb_pop_shows,
    top20_tmdb_pop_shows["tmdb_popularity"],
    "Top 20 most popular AppleTV shows on TMDB",
    "Show name",
    "Popularity",
    12,
    8,
    rotation=True,
    palette="autumn",
    plot_num=19,
)
merg = top20_tmdb_pop_shows.merge(shows[["title", "tmdb_score"]], on="title")
# no null values
merg.isnull().sum()
x = merg["title"]
y1 = merg["tmdb_popularity"]
y2 = merg["tmdb_score"]
fig, ax1 = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(15)
ax2 = ax1.twinx()

ax1.bar(x, y1, color="#453750")
ax2.plot(x, y2, "b-", marker="o", markersize=10, markerfacecolor="red")

plt.title(
    "Top 20 most popular movies with ratings on TMDB",
    color="black",
    fontsize=TITLE_FONT,
)
ax1.set_ylabel("Popularity", color="#453750", fontsize=LABEL_FONT)
ax2.set_ylabel("Ratings", color="blue", fontsize=LABEL_FONT)

ax1.set_xticklabels(x, rotation=90, ha="right", color="black", fontsize=TICK_FONT)


c = 0

for p in ax1.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax1.annotate(
        "{:}".format(round(merg["tmdb_popularity"][c])),
        (x.mean(), y),
        ha="center",
        va="bottom",
        color="black",
        fontsize=15,
    )  # set the alignment of the text
    c += 1
# save the plot
plt.savefig("fig19.png", bbox_inches="tight")
## ECHO 3 IS THE MOST POPULAR SHOW ON TMDB BUT WITH LOW RATING ##


# distribution year 2000-2023
release_year_count = titles["release_year"].value_counts()
release_year_count = pd.DataFrame(release_year_count)

plt.figure(figsize=(10, 9))
sns.lineplot(data=release_year_count)
plt.title("AppleTV´s shows and movies release date (2000-2023)", fontsize=TITLE_FONT)
plt.xlim(2000, 2021)
plt.xlabel("Year released", fontsize=LABEL_FONT)
plt.ylabel("Total Shows on AppleTV", fontsize=LABEL_FONT)
# save the plot
plt.savefig("fig20.png", bbox_inches="tight")
## APPLETV HAS PRODUCED ONLY 1 MOVIE/SHOW FROM 2000-2018. BUT IT STARTS TO PRODUCE A LOT ##
## OF MOVIE/SHOWS FROM 2019-2023

df = (
    titles.groupby("release_year")["release_year"]
    .agg("count")
    .sort_values(ascending=False)
    .head()
    .reset_index(name="count")
    .sort_values(by="release_year", ascending=True)
    .reset_index()
)
x = df["release_year"]
y = df["count"]
order = np.arange(2019, 2024)
order = order

bar_plot2(
    x,
    y,
    df,
    df["count"],
    "Release year distrbution (2019 - 2023)",
    "Year",
    "Count",
    12,
    8,
    order=order,
    plot_num=21,
)

df = titles[
    (titles["release_year"] == 2019)
    | (titles["release_year"] == 2020)
    | (titles["release_year"] == 2021)
    | (titles["release_year"] == 2022)
    | (titles["release_year"] == 2023)
]
df = df.groupby("release_year")[["imdb_score", "tmdb_score"]].agg("mean").reset_index()
x = df["release_year"]
y1 = df["imdb_score"]
y2 = df["tmdb_score"]
bar_plot(
    x,
    y1,
    df,
    df["imdb_score"],
    "Average AppleTV movies and shows score on IMDB",
    "Year",
    "Score",
    12,
    8,
    rotation=True,
    palette="winter",
    plot_num=22,
)
bar_plot(
    x,
    y2,
    df,
    df["tmdb_score"],
    "Average AppleTV movies and shows score on TMDB",
    "Year",
    "Score",
    12,
    8,
    rotation=True,
    palette="flare",
    plot_num=23,
)
## APPLETV PRODUCED MORE MOVIES/SHOWS ON 2023 (ACCORDING TO IMDB) ##
## APPLETV PRODUCED MORE MOVIES/SHOWS ON 2019-2020 (ACCORDING TO TMDB) ##

genres_df = pd.DataFrame(columns=titles.columns)
genres_df.drop(["seasons", "production_countries"], axis=1, inplace=True)
for i in titles.index:
    iD = titles["id"][i]
    title = titles["title"][i]
    t = titles["type"][i]
    desc = titles["description"][i]
    release_year = titles["release_year"][i]
    age_cer = titles["age_certification"][i]
    runtime = titles["runtime"][i]
    genres = literal_eval(titles["genres"][i])

    imdbid = titles["imdb_id"][i]
    imdb_score = titles["imdb_score"][i]
    imdb_votes = titles["imdb_votes"][i]
    tmdb_pop = titles["tmdb_popularity"][i]
    tmdb_score = titles["tmdb_score"][i]

    for j in genres:
        genres_df.loc[len(genres_df.index)] = [
            iD,
            title,
            t,
            desc,
            release_year,
            age_cer,
            runtime,
            j,
            imdbid,
            imdb_score,
            imdb_votes,
            tmdb_pop,
            tmdb_score,
        ]
x = genres_df["genres"]
order = x.value_counts().index
count_plot(
    x,
    genres_df,
    "AppleTV movies and shows distrbution",
    "Genre",
    "Count",
    12,
    8,
    order=order,
    rotation=True,
    palette="flare",
    plot_num=24,
)
## APPLE PRODUCES MORE MOVIES AND SHOWS OF THE DRAMA AND COMEDY GENRE

movies_genre = genres_df[genres_df["type"] == "MOVIE"]
shows_genre = genres_df[genres_df["type"] == "SHOW"]
# filter comedy genre
filter_data = movies_genre[movies_genre["genres"] == "comedy"]
filter_data2 = shows_genre[shows_genre["genres"] == "comedy"]

gp = (
    filter_data.groupby("title")["imdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
gp2 = (
    filter_data2.groupby("title")["imdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
# top rated comedy movies on IMDB
x = gp["title"]
y = gp["imdb_score"]
bar_plot(
    x,
    y,
    gp,
    gp["imdb_score"],
    "Top rated comedy movies on IMDB",
    "Comedy movie name",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="summer",
    plot_num=25,
)
# top rated comedy shows on IMDB
x2 = gp2["title"]
y2 = gp2["imdb_score"]
bar_plot(
    x2,
    y2,
    gp2,
    gp2["imdb_score"],
    "Top rated comedy shows on IMDB",
    "Comedy show name",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="autumn",
    plot_num=26,
)

gp = (
    filter_data.groupby("title")["tmdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
gp2 = (
    filter_data2.groupby("title")["tmdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
# top rated comedy movies on TMDB
x = gp["title"]
y = gp["tmdb_score"]
bar_plot(
    x,
    y,
    gp,
    gp["tmdb_score"],
    "Top rated comedy movies on TMDB",
    "Comedy movie name",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="winter",
    plot_num=27,
)
# top rated comedy shows on TMDB
x2 = gp2["title"]
y2 = gp2["tmdb_score"]
bar_plot(
    x2,
    y2,
    gp2,
    gp2["tmdb_score"],
    "Top rated comedy shows on TMDB",
    "Comedy show name",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="flare",
    plot_num=28,
)

# filter drama genre
filter_data = movies_genre[movies_genre["genres"] == "drama"]
filter_data2 = shows_genre[shows_genre["genres"] == "drama"]

gp = (
    filter_data.groupby("title")["imdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
gp2 = (
    filter_data2.groupby("title")["imdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
# top rated drama movies on IMDB
x = gp["title"]
y = gp["imdb_score"]
bar_plot(
    x,
    y,
    gp,
    gp["imdb_score"],
    "Top rated drama movies on IMDB",
    "Drama movie name",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="summer",
    plot_num=29,
)
# top rated drama shows  on IMDB
x2 = gp2["title"]
y2 = gp2["imdb_score"]
bar_plot(
    x2,
    y2,
    gp2,
    gp2["imdb_score"],
    "Top rated drama shows on IMDB",
    "Drama show name",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="autumn",
    plot_num=30,
)


gp = (
    filter_data.groupby("title")["tmdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
gp2 = (
    filter_data2.groupby("title")["tmdb_score"]
    .agg(sum)
    .sort_values(ascending=False)
    .reset_index()
)
# top rated drama movies on TMDB
x = gp["title"]
y = gp["tmdb_score"]
bar_plot(
    x,
    y,
    gp,
    gp["tmdb_score"],
    "Top rated drama movies on TMDB",
    "Drama movie name",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="deep",
    plot_num=31,
)
# top rated drama shows on TMDB
x2 = gp2["title"]
y2 = gp2["tmdb_score"]
bar_plot(
    x2,
    y2,
    gp2,
    gp2["tmdb_score"],
    "Top rated drama shows on TMDB",
    "Drama show name",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="pastel",
    plot_num=32,
)


df = (
    genres_df.groupby("genres")["imdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
)
df2 = (
    genres_df.groupby("genres")["tmdb_score"]
    .agg("sum")
    .sort_values(ascending=False)
    .reset_index()
)
# distribution by genre on IMDB
x = df["genres"]
y = df["imdb_score"]
bar_plot(
    x,
    y,
    df,
    df["imdb_score"],
    "Distribution IMDB Score by Generes",
    "Genere",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="muted",
    plot_num=33,
)
# distribution by genre on TMDB
x2 = df2["genres"]
y2 = df2["tmdb_score"]
bar_plot(
    x2,
    y2,
    df2,
    df2["tmdb_score"],
    "Distribution TMDB Score by Generes",
    "Genere",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="flare",
    plot_num=34,
)

# average IMDB and TMDB scores by genre
df = (
    genres_df.groupby("genres")["imdb_score"]
    .agg("mean")
    .sort_values(ascending=False)
    .reset_index()
)
df2 = (
    genres_df.groupby("genres")["tmdb_score"]
    .agg("mean")
    .sort_values(ascending=False)
    .reset_index()
)

x = df["genres"]
y = df["imdb_score"]
bar_plot(
    x,
    y,
    df,
    df["imdb_score"],
    "Average IMDB Score by Generes",
    "Genere",
    "IMDB score",
    12,
    8,
    rotation=True,
    palette="bright",
    plot_num=35,
)

x2 = df2["genres"]
y2 = df2["tmdb_score"]
bar_plot(
    x2,
    y2,
    df2,
    df2["tmdb_score"],
    "Average TMDB Score by Generes",
    "Genere",
    "TMDB score",
    12,
    8,
    rotation=True,
    palette="dark",
    plot_num=36,
)

# production countries
production_countries_df = pd.DataFrame(columns=titles.columns)
production_countries_df.drop(["seasons", "genres"], axis=1, inplace=True)
for i in titles.index:
    iD = titles["id"][i]
    title = titles["title"][i]
    t = titles["type"][i]
    desc = titles["description"][i]
    release_year = titles["release_year"][i]
    age_cer = titles["age_certification"][i]
    runtime = titles["runtime"][i]
    production_countries = literal_eval(titles["production_countries"][i])
    imdbid = titles["imdb_id"][i]
    imdb_score = titles["imdb_score"][i]
    imdb_votes = titles["imdb_votes"][i]
    tmdb_pop = titles["tmdb_popularity"][i]
    tmdb_score = titles["tmdb_score"][i]

    for j in production_countries:
        production_countries_df.loc[len(production_countries_df.index)] = [
            iD,
            title,
            t,
            desc,
            release_year,
            age_cer,
            runtime,
            j,
            imdbid,
            imdb_score,
            imdb_votes,
            tmdb_pop,
            tmdb_score,
        ]


x = production_countries_df["production_countries"].value_counts().index
y = production_countries_df["production_countries"].value_counts().values
bar_plot(
    x,
    y,
    production_countries_df,
    y,
    "Distrbution of Production Countries",
    "Country",
    "Count",
    12,
    8,
    plot_num=37,
)

country = production_countries_df["production_countries"].value_counts()
fig = px.choropleth(
    locations=country.index,
    color=country.values,
    color_continuous_scale=px.colors.sequential.Reds,
    template="plotly_white",
    title="Distribution of Production by Countries",
)
# set the layout of the map
fig.update_layout(font=dict(size=TITLE_FONT, family="Arial"))
# save the map as a png image
fig.write_image("distribution_country.png", engine="kaleido", format="png")
