# Movies_ETL

## Challenge
For this task, assume that the updated data will stay in the same formats: Wikipedia data in JSON format and Kaggle metadata and rating data in CSV formats.

1. Create a function that takes in three arguments:
- Wikipedia data
- Kaggle metadata
- MovieLens rating data (from Kaggle)
2. Use the code from your Jupyter Notebook so that the function performs all of the transformation steps. Remove any exploratory data analysis and redundant code.
3. Add the load steps from the Jupyter Notebook to the function. You’ll need to remove the existing data from SQL, but keep the empty tables.
4. Check that the function works correctly on the current Wikipedia and Kaggle data.
5. Document any assumptions that are being made. Use try-except blocks to account for unforeseen problems that may arise with new data.

### Final Code
[config.py.zip](https://github.com/efuen0077/Movies-ETL/files/4572613/config.py.zip)

### Analysis

After extracting, transforming, and loading the data so that it is readable for the Hackathon participants, quite a few analyses can be performed with this clean data. One can view the data in SQL to determine which movies would be the most popular according to the ratings.
