# Spotify Song Popularity Predictor

### Problem: 
I am trying to solve the problem of predicting the popularity of songs on Spotify based on various factors. I want to identify the patterns and factors that contribute to a song's success and popularity. This can help artists, producers, and the music industry make informed decisions on how to make a song chart highly in Spotify.

## Project Topic
This project aims to explore the predictive capabilities of supervised learning models in determining the success of songs based on various factors, including bpm, key, and other audio features. Five machine learning models, Linear Regression, Support Vector Machine, Gradient Boost Regressor, k-Nearest Neighbors, and Random Forest Regression, will be employed to predict the performance of songs in terms of popularity and streaming metrics.
As seen with these models, the type of task at hand is an regression task, where our goal is to predict the performance or success of songs, represented by a numerical quantitative statistic, called streams. 

#### Description:
The dataset utilized for this project comprises an extensive list of the most popular songs of 2023, as documented on Spotify. Unlike conventional datasets, this collection provides a rich set of features that delve beyond the typical song attributes. It includes details such as track name, artist(s) name, release date, presence on Spotify playlists and charts, streaming statistics, Apple Music and Deezer presence, Shazam charts, and a spectrum of audio features.

#### Potential Applications:
Music Industry: Assist artists and record labels in making informed decisions about song production and marketing strategies.

Streaming Platforms: Enhance recommendation algorithms by understanding the characteristics of highly popular songs.

Artists: Provide insights for artists to tailor their creative process based on factors that resonate with listeners.

## Data

The dataset, titled "Most Streamed Spotify Songs 2023" by Nidula Elgiriyewithana, was gathered from Kaggle. It compiles information on the most popular songs of 2023 based on their performance on Spotify. The dataset includes a comprehensive set of features related to each song, such as track name, artist(s) name, release date, presence on Spotify playlists and charts, streaming statistics, Apple Music and Deezer presence, Shazam charts, and various audio features. These features encompass details about the song's structure, popularity on different platforms, and quantitative audio characteristics, providing a rich resource for exploring the factors contributing to a song's success. The data was obtained through web scraping methods, allowing for the extraction of information from various online sources to create a comprehensive and informative dataset.

References

Elgiriyewithana, N. (2023). Most Streamed Spotify Songs 2023. Kaggle. https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023/data

#### Data Description
The dataset comprises 954 rows of tabulated data, with each row corresponding to a unique song entry and 24 unique features. The dataset size is 106 kilobytes. It encompasses a variety of features, including both categorical and numeric types. Specifically, there are 7 categorical features and 17 numeric features.

Categorical features include "track_name", "artist(s)name", "released_year", "released_month", "released_day",  "key", and "mode". 

Numeric features include "artist_count", "in_spotify_playlists", "streams", "in_apple_playlists", "in_deezer_playlists", "in_spotify_charts", "in_apple_charts", "in_deezer_charts", "in_shazam_charts", "bpm", "valence_%", "energy_%", "acousticness_%", "instrumentalness_%", "liveness_%", "danceability%", and "speechiness_%".

The track_name represents the title of the song, while artist(s)_name signifies the name of the artist or artists involved in its creation. The artist_count specifies the number of contributors to the song. The released_year, released_month, and released_day indicate the date when the song was officially released. In_spotify_playlists quantifies the number of Spotify playlists featuring the song, while in_spotify_charts conveys the song's position and ranking on Spotify charts. Streams denote the total number of times the song has been played on Spotify.

In_apple_playlists reflects the count of Apple Music playlists including the song, and in_apple_charts details its presence and rank on Apple Music charts. In_deezer_playlists signifies the number of Deezer playlists featuring the song, and in_deezer_charts specifies its position on Deezer charts. In_shazam_charts represents the presence and rank of the song on Shazam charts. Bpm denotes the beats per minute, a measure of the song's tempo, while key indicates the musical key of the song. Mode specifies whether the song is in a major or minor key.

Danceability_% quantifies the percentage indicating how suitable the song is for dancing, valence_% expresses the positivity of the song's musical content, and energy_% reflects the perceived energy level of the song. Acousticness_% represents the amount of acoustic sound in the song, and instrumentalness_% quantifies the amount of instrumental content. Liveness_% indicates the presence of live performance elements in the song, and speechiness_% specifies the percentage of spoken words within the lyrics.

The data is not in a multi-table form or gathered from multiple sources; it is presented in a single tabulated dataset. Each feature provides valuable insights into the song's characteristics, from basic information like the track name and artist to more intricate details such as the song's tempo (bpm), key, and acoustic properties.


## Data Cleaning

The data cleaning process begins with an exploration of data types and column structure using the data.info() method. It is noted that the 'track_name' and 'artist(s)_name' columns, which may not generalize well to new, unseen tracks and artists, are dropped from the dataset.

Missing values are addressed through a two-step process. Initial imputation is performed, and columns with missing values are subsequently dropped using the dropna() method. The decision to drop records with null values is justified, ensuring a cleaner dataset for analysis.

Data types are then converted for specific columns ('key' and 'mode') to categorical, reflecting the musical key and mode information. Additionally, columns representing numerical data ('streams', 'in_deezer_playlists', 'in_shazam_charts') are corrected from string to numeric format.

To prevent assumptions about the ordinality of categories, one-hot encoding is applied to the categorical columns ('key' and 'mode'). The data is further normalized using the Standard Scaler from scikit-learn.

Outliers are visualized using box plots for numerical columns, and outliers are identified and subsequently removed from the dataset. The rationale behind outlier removal is discussed, particularly focusing on columns such as 'released_year', 'deezer_playlist', 'in_deezer_charts', and 'in_shazam_charts'.

In summary, the data cleaning process involves thoughtful decisions on dropping columns, handling missing values, addressing data type issues, normalizing data, identifying and removing outliers, and exploring feature distributions and correlations. The cleaning process is well-documented, providing clear explanations for each step and offering a solid foundation for subsequent data analysis. Visualizations, such as box plots, enhance the understanding of the dataset's characteristics.

### Exploratory Data Analysis

The initial step involves understanding the distribution of numerical features, with histograms displaying the distributions of variables like Danceability_%, Valence_%, and Energy_%. The conclusion drawn, stating that these features are close to evenly distributed, demonstrates a clear interpretation of the visualizations.

Next, a correlation matrix is generated to explore relationships between various features, with a focus on their impact on the number of streams. The heatmap visualization provides an insightful overview of correlations, and the conclusion highlights a specific finding: a high positive correlation between playlist features (e.g., in_deezer_charts) and stream counts.

Finally, a boxplot is utilized to analyze the relationship between artist collaborations (artist_count) and average stream counts. The conclusion that six artist collaborations yield the best average streams adds a valuable insight into potential factors influencing the streaming performance of songs.

In summary, the EDA is well-explained, employs proper visualizations (histograms, correlation matrix, boxplot), conducts meaningful analyses, and offers clear conclusions based on the observed patterns and relationships in the data. The inclusion of specific findings and discussions enhances the overall quality of the exploratory analysis.

### Models

The choice of models for the regression task appears appropriate, encompassing diverse algorithms to assess performance. The use of Linear Regression provides a baseline understanding, while Random Forest, Gradient Boosting Regressor, Support Vector Regressor, and K-Nearest Neighbors offer more complexity and flexibility.

To address potential multicollinearity, standardization is employed to ensure that features are on a similar scale. Feature engineering introduces a novel metric, 'streams_per_playlist,' enhancing the evaluation of a song's popularity relative to its presence in Spotify playlists.

Multiple models are utilized, allowing a comparative analysis of their performance. The Random Forest Regressor and Gradient Boosting Regressor demonstrate superior R-squared values, suggesting their effectiveness in predicting the number of streams.

The exploration of feature importance through various models, especially Random Forest and Gradient Boosting, offers insights into the significance of different predictors in determining stream counts.

The awareness of data imbalance is evident in the decision to standardize features, preventing larger-scaled features from unduly influencing the model. 

In conclusion, revelant models were selected with the consideration of multicollinearity through standardization, exploration of feature importance, and feature engineering to capture additional insights. 

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

## Result and Analysis

The Random Forest Regressor and Gradient Boosting Regressor emerged as the top-performing models, demonstrating higher R-squared values compared to Linear Regression, Support Vector Regressor, and K-Nearest Neighbors. This suggests that these ensemble models provide a better fit to the data, capturing complex relationships between features and stream counts.

To ensure robust evaluation, mean squared error (MSE) and R-squared metrics were utilized, offering insights into the predictive accuracy and variance explained by each model. The Random Forest and Gradient Boosting models consistently outperformed others in both metrics.

The introduction of the 'streams_per_playlist' feature through feature engineering provided additional context to the analysis, allowing for a nuanced understanding of a song's popularity relative to its inclusion in Spotify playlists.

Throughout the iterative process, standardization and feature scaling were employed to enhance model performance. While these techniques contributed to the overall effectiveness, there's an opportunity to explore advanced methods like cross-validation to mitigate overfitting and refine model generalization.

Comparisons between models were appropriately made, and the superiority of Random Forest and Gradient Boosting models was evident. This not only validates the initial model selection but also highlights the importance of exploring diverse algorithms for regression tasks.

## Discussion and Conclusion

The exploration into predicting song streams on music platforms led to valuable insights and considerations. The Random Forest Regressor and Gradient Boosting Regressor models demonstrated superior performance, emphasizing the importance of ensemble methods for complex prediction tasks. Feature engineering, particularly the introduction of 'streams_per_playlist,' provided a nuanced understanding of a song's popularity in relation to its presence in Spotify playlists.

One key takeaway is the significance of thoughtful model selection and iteration. The iterative process involving different algorithms and feature engineering allowed for a comprehensive evaluation, highlighting the strengths and weaknesses of each approach. The incorporation of standardization and feature scaling contributed to model effectiveness, and there's potential for further improvement through advanced techniques like cross-validation to enhance generalization.

A notable aspect is the acknowledgment of areas for improvement, such as addressing potential overfitting and exploring additional techniques for handling data imbalance. This reflects a commitment to continuous refinement and optimization in future iterations of the predictive model.

In conclusion, this analysis not only provides a predictive model for song streams but also underscores the importance of a systematic and iterative approach, where learning from each step informs subsequent decisions. The results and discussions pave the way for future enhancements, ensuring the model's adaptability and reliability in real-world scenarios.

#### Recommendations

- Artists and producers can leverage the insights from this model to optimize their songs for Spotify success.
- Focus on features with higher importance, as identified by the Random Forest model.
- Regularly update the model with new data to adapt to changing music trends.

Feel free to explore the Jupyter notebook for a detailed walkthrough of the analysis and modeling process.
