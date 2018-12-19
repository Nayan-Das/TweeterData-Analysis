Important Note : Please do not run the code on a low resource system!

Classifiers will take a considerable amount of time to generate the results owing to massive size of the data.

Run the following command in a linux terminal to download the dataset and codes for this project

'git clone https://soumyabanerjee1996@bitbucket.org/soumyabanerjee1996/cs685a-data-mining-term-project-group-08.git'

It will download the folder named 'cs685a-data-mining-term-project-group-08' which will contain 'project.zip'. Extract the zip to find the datasets and code.  To run the codes, the following python libraries are required to run the codes :

python - 3.6

jupyter-notebook

scikit-Learn

nltk with nltk.data downloaded

numpy

pandas

json

tweepy

unicodedata

scipy

matplotlib

wordcloud

Description of each files of every folder are given below : 

==========================================================================================
dataset folder:
    
    crawler.py - it is used to download tweets from twitter
    
    test.json - it contains test dataset in json format 
    
    training.json - it contains training dataset in json format
    
    NepalQuake-test-46K-tweetids.txt - it contains 46k tweet ids for test set which are downloaded and stored in test.json
    
    NepalQuake-training-20K-tweetids.txt - it contains 20k tweet ids for training set which are downloaded and stored in training.json
    
    NepalQuake-test-availability-tweetids.txt - contains ground truth for available tweet id in the test set
    
    NepalQuake-test-need-tweetids.txt - contains ground truth for need tweet id in the test set
    
    NepalQuake-training-availability-tweetids.txt - contains ground truth for available tweet id in the training set
    
    NepalQuake-training-need-tweetids.txt - contains ground truth for need tweet id in the training set


==========================================================================================    
    
1_gram_tf_idf_non_under_sample_data folder:

****hindi_to_english folder:
    
        y_train_creator.ipynb - contains code to create label for training set using availability tweet id and need tweet id files and 'y_train_creator.ipynb' is also used to create class label for test set also
        
        y_train_class_label.txt - contains class label for training set data
        
        y_test_class_label.txt - contains class label for test set data
        
        translator.ipynb - contains code to translate tweets to english using googletrans api and extract tweets from .json files
        
        X_test_features_sparse_matrix.npz and X_train_features_sparse_matrix.npz - contains feature matrix created using tf-idf vectorizer for test set and training set respectively
        
        processed_training_data.txt and processed_test_data.txt - containing processed tweets for training set and test set respectively
        
        test_data_id.txt and train_data_id.txt - contains tweet ids for test data set and training data set
        
        test_hindi_to_english_40974.txt and training_hindi_to_english_16932.txt - contains extracted tweets from test.json and train.json files
        
        preprocessing.ipynb - code used for preprocessing work
        
        feature_matrix_creation.ipynb - code to create feature matrix for test and training dataset
        
        datacloud.ipynb - code to create datacloud
        
        classifier.ipynb - code to run different classifiers


****working_on_hindi folder:
    
        y_train_creator.ipynb - contains code to create label for training set using availability tweet id and need tweet id files and 'y_train_creator.ipynb' is also used to create class label for test set also
        
        translator.ipynb - contains code to extract tweets from .json files
        
        y_train_class_label.txt - contains class label for training set data
        
        y_test_class_label.txt - contains class label for test set data
        
        X_test_features_sparse_matrix.npz and X_train_features_sparse_matrix.npz - contains feature matrix created using tf-idf vectorizer for test set and training set respectively
        
        processed_training_data.txt and processed_test_data.txt - containing processed tweets for training set and test set respectively
        
        test_data_id.txt and train_data_id.txt - contains tweet ids for test data set and training data set
        
        test_natural_40974.txt and training_natural_16932.txt - contains extracted tweets from test.json and train.json files
        
        preprocessing.ipynb - code used for preprocessing work
        
        feature_matrix_creation.ipynb - code to create feature matrix for test and training dataset
        
        datacloud.ipynb - code to create datacloud
        
        classifier.ipynb - code to run different classifiers
        
****undersampling_hindi_to_english folder:

        y_train_creator.ipynb - contains code to create label for training set using availability tweet id and need tweet id files and 'y_train_creator.ipynb' is also used to create class label for test set also
        
        y_train_class_label.txt - contains class label for training set data
        
        y_test_class_label.txt - contains class label for test set data
        
        translator.ipynb - contains code to translate tweets to english using googletrans api and extract tweets from .json files
        
        X_test_features_sparse_matrix.npz and X_train_features_sparse_matrix.npz - contains feature matrix created using tf-idf vectorizer for test set and training set respectively
        
        processed_training_data.txt and processed_test_data.txt - containing processed tweets for training set and test set respectively
        
        test_data_id.txt and undersampled_train_id.txt - contains tweet ids for test data set and training data set
        
        test_hindi_to_english_40974.txt and training_hindi_to_english_16932.txt - contains extracted tweets from test.json and train.json files
        
        preprocessing.ipynb - code used for preprocessing work
        
        feature_matrix_creation.ipynb - code to create feature matrix for test and training dataset
        
        classifier.ipynb - code to run different classifiers
        
        undersamling.ipynb - contains code to undersample training data and only the irrelevant class is undersampled
        
****undersampling_working_on_hindi folder:

        y_train_creator.ipynb - contains code to create label for training set using availability tweet id and need tweet id files and 'y_train_creator.ipynb' is also used to create class label for test set also
        
        translator.ipynb - contains code to extract tweets from .json files
        
        y_train_class_label.txt - contains class label for training set data
        
        y_test_class_label.txt - contains class label for test set data
        
        X_test_features_sparse_matrix.npz and X_train_features_sparse_matrix.npz - contains feature matrix created using tf-idf vectorizer for test set and training set respectively
        
        processed_training_data.txt and processed_test_data.txt - containing processed tweets for training set and test set respectively
        
        test_data_id.txt and undersampled_train_id.txt - contains tweet ids for test data set and training data set
        
        test_natural_40974.txt and training_natural_16932.txt - contains extracted tweets from test.json and train.json files
        
        preprocessing.ipynb - code used for preprocessing work
        
        feature_matrix_creation.ipynb - code to create feature matrix for test and training dataset
        
        classifier.ipynb - code to run different classifiers
        
        undersamling.ipynb - contains code to undersample training data and only the irrelevant class is undersampled
        
==========================================================================================    
    
3_gram_tf_idf_non_under_sample_data folder:

    It follows same rules as 1_gram_tf_idf_non_under_sample_data folder
