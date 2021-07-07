### Capstone-Project-University-of-Texas

#### CHATBOT INTERFACE-TEXT CLASSIFICATION USING NATURAL LANGUAGE PROCESSING

Problem Statement
The Capstone project is designed to undertake a multi-faceted project that demonstrates our
understanding and mastery of the key conceptual and technological aspects of Deep Learning
and targets to develop an understanding of how challenging human-level problems can be
approached and solved using a combination of tools and techniques. The database comes from
one of the biggest industries in Brazil and in the world. It is an urgent need for industries /
companies around the globe to understand why employees still suffer some injuries / accidents in
plants. Sometimes they also die in such an environment. Considering Industrial safety, we need
to analyze the risk of accidents.

Objective
This Project aims at designing a Machine Learning, Deep Learning and Natural Language
Processing based chatbot utility which can help the professionals to highlight the safety risk as
per the incident description. The Chatbot helps technicians by automating the interaction process
to get access for immediate help and support, avoiding the need for manual interaction.

NLP Preprocessing
The NLP preprocessing was performed on an incident description feature. Removal of
stopwords, lemmatization and tokenization, data vectorization of the text was performed to use it
for our modeling purpose
Lemmatization: An instance of the WordNetLemmatizer() is created and call the lemmatize()
function on a single word
Activation to Activate
Removing to Remove
Count Vectorization: Count Vectorization will create a matrix consisting of term/token counts by
the number of rows.
TFIDF Vectorization: It is an information retrieval technique that weights a term's frequency TF
and its inverse document frequency IDF.Based on the occurrence of a term in a document, it will
have its own TF and IDF score.A product of these two scores gives us the weight of the term in
the document and the higher the score is, the rarer the term is in the given document and
vice-versa

Design, Train and Test Machine Learning Classifiers
Since we need to create a classification model we started with ML models to create a benchmark
for our future model training. The data was trained using the below Machine Learning
algorithms:
1. LightGBMClassifier
2. RandomForest Classifier
3. XGBOOST Classifier
These models were selected since these are ensemble techniques which can make better
predictions and achieve better performance compared to other classification models


Design, Train and Test Neural Networks Classifier
The data was trained using a neural network using a sequential model. We built a traditional NN
model to classify the text description into different critical risks.
Model Structure:
● Training begins by calling the fit () method.
● Dense layer is defined with input and output size.
● Only the first layer of the model requires the input dimension to be explicitly stated and the following
layers are able to infer from the previous linear stacked layer.
● Following standard practice, the rectified linear unit activation function is used for this layer.
● Dropout is used to reduce overfitting for regularization. Since it is a multi-class classification problem we
are solving with our network, the activation function for the last layer is set to softmax.● The learning rate is a hyperparameter which determines to what extent newly acquired weights overrides
old weights. In general it lies between 0 and 1. Here we set the learning rates for neural networks to 0.1.
● Momentum optimization is used to decide the weight on nodes from previous iterations. It helps in
improving training speed and also in avoiding local minimas. Here, we set momentum to 0.85.
● Gradient descent is a method that defines a cost function of parameters and uses a systematic approach to
optimize the values of parameters to get the minimum cost function. Here we use Stochastic gradient
descent that updates the parameters by calculating gradients for each training example.
● The number of epochs is set to 100
In general, error/loss for a neural network is the difference between actual value and predicted
value. The goal is to minimize the error/loss. Loss Function is a function that is used to calculate
the error. Here we used categorical_crossentropy since it is a multiclass problem.
Predictions are done and the classification report is generated. The model results were
satisfactory but we decided to increase the model accuracy by using LSTM Model


Design, Train and Test LSTM Classifier
LSTM network models are a type of recurrent neural network that are able to learn and
remember over long sequences of input data. Since LSTMs are suited for classification, they may
be a good fit for this problem.
Model Structure
● LSTM classifier is built with initializing the dataset with top 50000 words and set the maximum number of
words to 250 and set the embedding the dimension to 100.
● Tokenizer is used to tokenize and transform the words into numerical data.
● Pad_sequences is used to pad the data to make it of same length(max_length)
● The Categorical labels (Critical risk) is converted to numbers using pandas get_dummies.
● The first layer is the embedded layer that uses 100 length vectors to represent each word.
● SpatialDropout1D performs variational dropout in NLP models.
● The next layer is the LSTM layer with 100 memory units.
● The output layer must create 13 output values, one for each class.
● Activation function is softmax for multi-class classification.
● Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.
In optimization, the main aim is to find weights that reduce loss. Adam optimizer is one of the
most effective optimization algorithms for training neural networks. It combines ideas from
RMSProp and Momentum.
The model is fit and evaluated. The loss and accuracy is found from model.evaluate. The results
are plotted and a classification report is generated


![ss1](https://user-images.githubusercontent.com/67209958/124840599-37ae4180-df40-11eb-86de-60b40d7dd453.JPG)
