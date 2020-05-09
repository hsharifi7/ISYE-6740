Jason Joseph Rebello
Carnegie Mellon University (Jan 2012 - May 2013)
Masters in Electrical & Computer Engineering
Logistic Regression with Regularization

Run LogisticRegression.m
This program uses Logistic Regression to classify handwritten digits. The data is comprised of a part of the MNIST dataset. 5000 samples are used

First a part of the data set is shown in order to visualize what we are dealing with. In practice it is always good to first look at the data before deciding what algorithm to use.

The data is then split into Training and Test sets. Since we have a total of 5000 samples (500 belonging to each label 0-10), I use 80% of the total for training i.e 4000 samples for training and 1000 for testing. The way this is done is, 500 samples belong to each class, out of which 400 from each class are used for training and 100 from each class are used for testing. Out of the 500 in each class, 400 are randomly chosen by using the randperm function and the remaining 100 are used for testing.

The training data is then passed to the Logistic Regression Classifier in order to train the model. Here for each label we analyze the 400 samples belonging to that particular label and build a classifier via the one-vs-all method. The cost function is calculated in order to insure that gradient descent is running properly.

The cost function is implemented with regularization. Note: theta0 should not be regularized. Also, gradient descent for theta0 does not include regularization term as compared to the remaining thetas. fmincg performs gradient descent for 150 iterations. This can be changed in options parameter in the LRClassifier function.

Predictions are made on the test set by finding the max value for each classifier. The accuracy on the test set is then calculated. The accuracy will keep changing with each run of the program since the training and testing set is randomly chosen.