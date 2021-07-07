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

Model Finalization

● The LSTM model has the highest combined accuracy on train and validation dataset
● The LSTM model also performs best in terms of precision, recall & F1 score. It also
doesn't have dependency on the bag of words present in the dataset.
● Also as seen in the comparison of confusion matrix the LSTM classification results are
better than other models

Saving Model

Python pickle module is used for serializing and de-serializing python object structures. Used to
save complicated data. It is easy to use, lighter and doesn’t require several lines of code.The
pickled file generated is not easily readable and thus provide some security.We pickle the LSTM
model to use it in flask application.
We saved the model and its weight as an .h5 file. The file was used in the backend of chatbot to
make the predictions and also for future updates / training.

Integration of Model to User Interface

We need to integrate our model to UI for the users to use the chatbot to determine the critical risk
for the accident description provided by the user.

Creating Web Application using Flask

Flask web application was used for UI.
To install flask: pip install flask
We need to create 2 files: python file app.py (say) and an index.html file
app.py file: python file where we import the necessary libraries, create flask app,load pickle file.
The render_template ('index.html') looks for a file called index.html in the templates folder.
The /predict is a post method where we pass the features to the model.pkl file so that the model
will take the input and give output.
We pass the description field and perform cleaning, preprocessing,tokenizer followed by pad
sequences to it. Request.form.values takes input from all text fields and store it in a feature
called int_features.Give this to model as model.predict
A function to predict the description is defined that predicts the critical risk along with the
accident level and the potential accident level. If the description is out of range of the trained
text, then it displays the message "Sorry, I didn't get that.", "Sorry! I don't have an answer for
that."
This will return the prediction to the index.html page which outputs the prediction.
The application will run locally on the URL. Open a browser and type in the URL to run the
application.
Index.html file: The styles of output template is defined – background color, text alignment,font
size, postion,padding, border radius, font family, border radius is defined.
It starts with greeting note followed by getting the description as input.
It provides the critical risk category with the accident level and potential accident level and
prompts the user if they still need any help with other descriptions. If another description is
provided by the user, then it provides the output until the user replies back with bye


![ss1](https://user-images.githubusercontent.com/67209958/124840599-37ae4180-df40-11eb-86de-60b40d7dd453.JPG)
