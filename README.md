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
    I explore the dataset by checking its information, printing the first and last few records, checking for missing values, and analyzing the data per column values like counts of males/females, counts of ad clicks, and counts of ad clicks per country.

<h2>Visual Analysis of the Dataset</h2> 
    I create visualizations using seaborn and plotly to gain insights into the data, such as the count of male and female users clicking on the ad and the distribution of ads clicked per country.

<h2>Data Preprocessing</h2>
     Remove columns that are not required for building the ANN model, such as 'Country', 'Ad Topic Line', 'City', and 'Timestamp'.

<h2>Preparing Data for Modeling</h2>
     Split the data into input (predictors) and output (response) variables, and then further split the data into training and testing sets using the train_test_split function from sklearn.

<h2>Training the Neural Network</h2> 
    Build an ANN model using Keras, a high-level neural networks API running on top of TensorFlow. The model consists of three layers: an input layer with 64 neurons, a hidden layer with 8 neurons, and an output layer with 1 neuron using the sigmoid activation function for binary classification. Compile the model using binary cross-entropy as the loss function and Adam optimizer, and then train the model on the training data for 150 epochs.

<h2>Model Evaluation</h2>
     Finally, evaluate the trained model's performance using accuracy as the metric and print the classification report, confusion matrix, and accuracy score to assess how well the model predicts ad clicks based on the given features.

<h2>Conclusion</h2>
     

An artificial neural net is an information processing paradigm whose working is similar to the working of biological nervous systems. The key element of an artificial neural network is the novel structure of its information processing system. This structure consists of a large number of highly interconnected processing computing elements that work in unison to solve specific problems.

Few of the benifits of ANN are: 1.Ability to generalize their inputs. 2.Potential for high fault tolerance. When these networks are scaled across multiple machines and multiple servers, they are able to route around missing data or servers and nodes that can't communicate 3.Can regenerate large amounts of data by inference and help in determining the node that is not working

We have used Keras deep learning library for modeling regression problem. It enables us to write powerful Neural Networks with a few lines of code and runs on Tensorflow backend. Learnt how to develop and evaluate neural network models, including:

    How to load data and develop a baseline model.
    How to lift performance using data preparation techniques like standardization.
    How to design and evaluate networks with different varying topologies on a problem.

This Model when trained on the train dataset and when tested on the test dataset gives us an accuracy of around 87% and 85% respectively.
