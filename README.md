# Bangla_Sentiment_analysis
This work is designed to predict the sentiment of a Bengali news article and analyse a collection of such predicted data.


Instructions for executing INFERENCE code for sentiment prediction:

Please use the ‘Bsentiment_infer_v2.ipynb’ file for this purpose. This code can be executed in Google colab and any local system or Linux server with a GPU. The code is not friendly to be executed on a CPU-only system due to the usage of pre trained DL model.

If running this code colab, please upload the model in the drive (the account from which colab will be used), as it’s directly accessed there. Alternatively, you may manually add the model to the colab session using file uploads and adjust the model path accordingly. Otherwise, for local systems, please download them from here and save the models in your local directory.

While running the code in colab, you can initially add a zip file containing news articles to initialise the ‘/content’ directory with a few JSON files. Then, you can add more JSON files for further testing using the files upload option while the program runs. Those incoming files will be processed at regular intervals. This interval can be set by the user.

Please follow the instructions given in the code file as comments to comment out unnecessary lines of code based on the system being used.


Description of CODE files:

‘Bengali_Data_Preparation.ipynb’: In this file, the NLP tools for the Bengali language are explored for data preprocessing and visualisation. The training dataset is prepared in this file, combining and preprocessing the existing datasets. Also, the news articles dataset is prepared as an individual JSON file for each article.
‘bsa_train2.py’: This file contains the code to finetune the BERT model for the downstream task of sentiment classification.
‘Bsa_test.ipynb’: This file is used to test the performance of the finetuned model in terms of performance metrics like accuracy, F1-score, etc.
‘Analysis_Visualise.ipynb’: This file creates all the analysis plots on the prediction files.
‘Bsentiment_infer_v2.ipynb’: This file contains the code for real-time inference as desired in the requirement. This file contains code to process each incoming news file, generate the prediction and store them in an output directory.


Description of DATA files:

‘news1.zip’: It contains all 7411 JSON files of news articles.
‘news_sample.zip’: contains 100 news articles selected from the above files to create a small test dataset.
‘news1_output.zip’: contains the predicted output files for each news article.
‘stopwords_bangla.xlsx’: It contains Bengali stopwords. This is used during text preprocessing.
‘train_v3.xlsx’: Training dataset for the model.
‘test_v3.xlsx’: dataset to test the performance of the finetuned model using performance metrics like F1, accuracy, etc.

