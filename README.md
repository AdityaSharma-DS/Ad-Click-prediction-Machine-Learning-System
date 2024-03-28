<h1>'Ad Click'- prediction using ANN</h1>
    This project demonstrates the process of using an Artificial Neural Network (ANN) to predict ad clicks based on user features and evaluates the model's performance in terms of accuracy and predictive capability.

<h2>Dataset</h2>
    The dataset "advertising-1.csv"  used in this project contains information about users' interactions with ads, including whether they clicked on the ad or not. Contains following details.
    Daily Time Spent on Site	Age	Area Income	Daily Internet Usage	Ad Topic Line	City	Male	Country	Timestamp Clicked on Ad
 
<h2>Libraries Used</h2>
    <b>Pandas:</b> Pandas is a powerful data manipulation library in Python. In this project, Pandas is used to read the dataset from a CSV file (read_csv function), explore the dataset by checking its information, printing the first and last few records, checking for missing values, and analyzing the data per column values like counts of males/females, counts of ad clicks, and counts of ad clicks per country. It helps in data cleaning, preprocessing, and initial data exploration.

    <b>NumPy:</b> NumPy is a fundamental package for numerical computations in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. In this project, NumPy is used alongside Pandas for data manipulation and numerical computations, such as handling arrays of data and performing calculations on them.

    <b>Matplotlib:</b> Matplotlib is a plotting library in Python used for creating static, interactive, and animated visualizations. In this project, Matplotlib is used for creating basic visualizations like histograms, bar charts, line plots, and scatter plots to visualize the distribution of data, relationships between variables, and other insights related to the dataset.

    <b>Seaborn:</b> Seaborn is a statistical data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. In this project, Seaborn is used for creating more advanced and aesthetically pleasing visualizations like count plots, box plots, and heatmaps. These visualizations help in gaining deeper insights into the data and relationships between different variables.

    <b>Plotly:</b> Plotly is an interactive visualization library that allows creating interactive plots and dashboards. It provides support for a wide range of plots, including scatter plots, bar charts, line plots, and 3D plots. In this project, Plotly is used to create interactive visualizations that can be embedded in web applications or notebooks, enhancing the data exploration and presentation capabilities.

    <b>Keras:</b> Keras is a high-level neural networks API that is built on top of other deep learning libraries such as TensorFlow and Theano. It provides a user-friendly interface for building and training neural network models. In this project, Keras is used for building the Artificial Neural Network (ANN) model, defining the layers, activation functions, loss functions, optimizer, and training the model on the given data.

    <b>Scikit-learn (sklearn):</b> Scikit-learn is a popular machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It includes various algorithms for classification, regression, clustering, dimensionality reduction, and model selection. In this project, sklearn is used for splitting the dataset into training and testing sets (train_test_split function), evaluating the model's performance using metrics such as accuracy, and generating classification reports and confusion matrices.

<h2>Visualizing the Dataset</h2> 
    We explore the dataset by checking its information, printing the first and last few records, checking for missing values, and analyzing the data per column values like counts of males/females, counts of ad clicks, and counts of ad clicks per country.

<h2>Visual Analysis of the Dataset</h2> 
    We create visualizations using seaborn and plotly to gain insights into the data, such as the count of male and female users clicking on the ad and the distribution of ads clicked per country.

<h2>Data Preprocessing</h2>
     We remove columns that are not required for building the ANN model, such as 'Country', 'Ad Topic Line', 'City', and 'Timestamp'.

<h2>Preparing Data for Modeling</h2>
     We split the data into input (predictors) and output (response) variables, and then further split the data into training and testing sets using the train_test_split function from sklearn.

<h2>Training the Neural Network</h2> 
    We build an ANN model using Keras, a high-level neural networks API running on top of TensorFlow. The model consists of three layers: an input layer with 64 neurons, a hidden layer with 8 neurons, and an output layer with 1 neuron using the sigmoid activation function for binary classification. We compile the model using binary cross-entropy as the loss function and Adam optimizer, and then train the model on the training data for 150 epochs.

<h2>Model Evaluation</h2>
     Finally, we evaluate the trained model's performance using accuracy as the metric and print the classification report, confusion matrix, and accuracy score to assess how well the model predicts ad clicks based on the given features.