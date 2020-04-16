# Horse Racing Data for Machine learning
Note: Unfortunately, after coming back to this project to make it a little more presentable, I have discovered that changes were made to the website I scraped the data from, thus rendering scraping.py and preprocessing.py a bit useless. I might come back and update it, but for now there's example data in the data folder which can be used for testing with sklearn models.

The goal of this project was to get familiar with python and data manipulation - as well as put some of the concepts I learnt in my university machine learning course to the test. Learning how to use python and specifically the pandas library ended up being a valuable resource.

# Usage
Clone the repository and install the requirements:
pip install -r requirements.txt
run model.py

The 'model' variable in model.py can be replaced with any [sklearn model types](https://scikit-learn.org/stable/supervised_learning.html)

# Practical things I learnt
- One-Hot Encoding
- Creating syntehetic variables
- Scaling data
- Problem of local minima
- Problem of class imbalance and ways to solve it (undersampling, oversampling, different ways to do each)
- Experience playing with the hyperparameters of several different model types

# Things to improve on
- Implement a model using tensorflow. sklearn is great for learning but really offers very little customisability. If this was ever to be possible it would need to be with a more bespoke model.
- Re-implement the scraping for new website layout
- Change the whole approach to this problem. The way in which individual horses respond to different features is different depending on the horse. If I were to tackle this again from scratch, I'd build a model for each horse (or perhaps for horses of the same dam or sire?) and predict a finishing time for each horse in the race based on individual factors. This would mean running into the problem of not having enough data for some horses, and would probably mean experimenting all over again with the different types of models.
