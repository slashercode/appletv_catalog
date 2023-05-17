# 'AppleTV+' Data Analysis and Visualization

## Project Overview

This repository contains information about a personal Data Science project aimed at performing data analysis and visualization of AppleTV+ Movies and TV Shows using Python. The data (**AppleTV+ Movies and TV Shows**) was collected from **Kaggle(source: JustWatch, March 2023)**.

The dataset contained information on variables such as title ID, title, show type, description, release year, age certification, runtime, genres, production countries, seasons, IMDB ID, IMDB score, IMDB votes, TMDb popularity and TMDB score. The goal of the project is to provide a comprehensive script that **_retrieves data_**, **_performs various analyses_**, and **_generates graphs to gain insights_** into the AppleTV+ content catalog. Additionally, a **_dashboard_** has been created using **Tableau** to present the findings in an **_interactive and visually appealing manner_**.

## Features

- Retrieves AppleTV+ Movies and TV Shows data.
- Performs data cleaning and preprocessing.
- Generates various graphs and visualizations.
- Creates a Tableau dashboard for interactive data exploration.

## Requirements

The following packages are required for this project:

- Pandas
- NumPy
- Seaborn
- Warnings
- Math
- Ast
- Missingno
- Matplotlib
- Plotly

It is recommended to install these packages using a package manager such as pip or conda. For example, to install these packages from the requirements.txt file using pip, open a terminal or command prompt and run the following command:

    pip install -r requirements.txt

**Note:** Some of these packages may have dependencies that need to be installed as well.

## Project Workflow

**Data Gathering:** Obtained the AppleTV+ movies and shows data from Kaggle.

**Data Preprocessing:** Loaded the data into Python using Pandas and perform necessary data cleaning and preprocessing steps. This involved handling missing values, obtaining the shape of the dataset, counting the number of null values, detecting the number of null values in the columns and filling the database with complete information.

**Data Analysis:** Performed various analyses on the AppleTV+ data to gain insights. This included calculating summary statistics, identifying popular genres, analyzing release patterns and examining user ratings on IMDB and TMDB. Use pandas functionalities, such as grouping, filtering, and aggregation, to conduct meaningful analyses.

**Data Visualization:** Utilized Matplotlib and Seaborn libraries to create informative and visually appealing graphs and visualizations. Generate plots such as bar charts, count plots, line charts to represent the findings from the data analysis phase.

**Tableau Dashboard:** Created a Tableau dashboard for interactive data exploration. Import the cleaned and preprocessed data into Tableau and design a visually engaging and user-friendly dashboard. Included relevant visualizations, filters, and interactive elements to allow users to explore the AppleTV+ data in a dynamic and intuitive manner.

## Conclusion

**Movies:** In terms of movies released on AppleTV+ in 2023, "The Italian Job" and "CODA" [According to the STATS in IMDB] and "Luck" and "Emancipation" [According to the STATS in TMDB] have garnered the most attention and viewership. These movies have resonated with the audience, likely due to their compelling narratives, strong performances, and positive critical reception. They have become the standout movies on the platform, capturing the interest of AppleTV+ subscribers.

**TV Shows:** Among the TV shows released on AppleTV+ in 2023, several have gained significant popularity. The analysis reveals that "Ted Lasso" and "Severence" [According to the STATS in IMDB] and "Echo 3" and "Ted Lasso" [According to the STATS in TMDB] have emerged as the top two most-watched TV shows during this period. These shows have attracted a large audience and received positive feedback, contributing to their popularity on the platform.

**Genres:** Analysis of the popular TV shows and movies on AppleTV+ indicates that certain genres have performed exceptionally well in 2023. Genres such as drama, comedy, family and animation have consistently attracted a significant viewership and have been key contributors to the platform's success.

**Viewer Engagement:** The analysis also highlights the high level of viewer engagement with the most popular content on AppleTV+. Through ratings, and reviews, it is evident that audiences have actively embraced and recommended these TV shows and movies, contributing to their popularity within the AppleTV+ community.

**_Note:_** These conclusions provide valuable insights into the preferences of AppleTV+ subscribers in 2023 and serve as a guide for understanding the most popular content on the platform during that period.

## Acknowledgement

You can view the dataset here: [AppleTV+ Movies and TV Shows 2023](https://www.kaggle.com/datasets/dgoenrique/apple-tv-movies-and-tv-shows)

## License

**NOT FOR COMMERCIAL USE**

_If you intend to use any of my code for commercial use please contact me and get my permission._

_If you intend to make money using any of my code please get my permission._
