### Predicting Spotify Song Popularity

Dataset located in Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023/data

#### Problem Statement

The goal of this project is to predict the popularity of songs on Spotify based on various factors. By identifying the patterns and factors that contribute to a song's success and popularity, this predictive model can assist artists, producers, and the music industry in making informed decisions to optimize a song's performance on Spotify.

#### Data

The dataset used for this project contains information about various features of songs, including audio characteristics, release details, and performance metrics on different platforms such as Spotify, Apple Music, Deezer, and Shazam.

#### Data Cleaning and Exploration

- Loaded the dataset, detected the encoding, and displayed the first few rows.
- Dropped irrelevant columns ('track_name' and 'artist(s)_name') that do not generalize well to new, unseen tracks and artists.
- Checked for missing values, dropped records with nulls, and performed data type conversions.
- Explored the relationship between musical key ('key') and streams, and between mode ('mode') and streams using visualizations.

#### Data Preprocessing

- Performed one-hot encoding on categorical columns ('key' and 'mode') to avoid assumptions about ordinality.
- Standardized numerical features to ensure that all features are on a similar scale.
- Identified and removed outliers in the dataset.

#### Feature Engineering

- Created a new feature, 'streams_per_playlist,' to measure a song's popularity relative to its presence in Spotify playlists. A higher ratio indicates more streams per playlist, suggesting higher engagement or interest from users.

#### Model Building and Evaluation

##### Linear Regression

- Utilized a linear regression model to predict song popularity.
- Evaluated the model using Mean Squared Error (MSE) and R-squared.

##### Random Forest

- Employed a Random Forest regression model for predicting song popularity.
- Assessed the model's performance using MSE and R-squared.

##### Support Vector Regressor (SVR)

- Applied SVR to predict song popularity.
- Standardized the input features and evaluated the model using MSE and R-squared.

##### K-Nearest Neighbors (KNN)

- Utilized KNN regression for predicting song popularity.
- Standardized the input features and assessed the model's performance using MSE and R-squared.

#### Model Comparison

- Linear Regression:
  - MSE: 0.3301
  - R-squared: 0.4807

- Random Forest:
  - MSE: 0.1880
  - R-squared: 0.7042

- SVR:
  - MSE: 0.3015
  - R-squared: 0.5258

- K-Nearest Neighbors (KNN):
  - MSE: 0.5712
  - R-squared: 0.1015

#### Conclusion

The Random Forest model demonstrated the best performance among the models considered, with the lowest MSE and the highest R-squared. This suggests that Random Forest is a suitable choice for predicting song popularity on Spotify based on the given features.

#### Recommendations

- Artists and producers can leverage the insights from this model to optimize their songs for Spotify success.
- Focus on features with higher importance, as identified by the Random Forest model.
- Regularly update the model with new data to adapt to changing music trends.

Feel free to explore the Jupyter notebook for a detailed walkthrough of the analysis and modeling process.
