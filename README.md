# CODTECH-TASK-4
COMPANY:CODTECH IT SOLUTIONS
NAME:VALLAPU SATHVIKA
INTERN ID:CT08RGJ
DOMAIN:PYTHON PROGRAMMING
DURATION:4 WEEKS
MENTOR:NEELA SANTOSH
**DESCRIPTION OF THE CODE***
This Python code implements a **spam email classification** model using a **Naive Bayes classifier**. It is performed within a **Jupyter Notebook**, where the code is executed step by step, allowing for easy debugging and visualization of results. The model is trained to distinguish between 'ham' (non-spam) and 'spam' messages based on a dataset, and evaluates its performance using several metrics such as accuracy, precision, recall, and confusion matrix visualization.

**Tools and Libraries Used**

1. **Pandas**: This library is used to handle and manipulate the dataset. It is used to load the CSV file and perform necessary data cleaning and preparation operations.
2. **NumPy**: This library is often used alongside Pandas for numerical operations and handling arrays.
3. **Scikit-learn**: This machine learning library provides several useful tools, such as:
   - **train_test_split**: Used for splitting the dataset into training and testing sets.
   - **TfidfVectorizer**: This tool converts text data into numerical form (using Term Frequency-Inverse Document Frequency), which is needed for machine learning models.
   - **MultinomialNB**: Implements the Naive Bayes algorithm, which is a simple yet effective probabilistic model for classification tasks, especially with text data.
   - **classification_report, accuracy_score, confusion_matrix**: These are performance evaluation tools that assess the model’s ability to classify messages accurately.
4. **Matplotlib**: This library is used for creating visualizations. It is used in this code to plot the confusion matrix, which is a crucial metric for classification tasks.

 **Step-by-Step Description of the Code**

1. **Loading the Dataset**:  
   The first step is loading the dataset using **Pandas**. The CSV file is read using `pd.read_csv()`, and the `encoding='latin-1'` is used to ensure that any special characters are correctly processed. If the file is not found, the program will terminate with an error message. After loading, the first few rows of the dataset are displayed using `df.head()`.

2. **Data Preprocessing**:  
   The dataset is expected to have two columns: `v1` (the label, either "ham" or "spam") and `v2` (the text message). If these columns are present, they are renamed to `label` and `message` for better clarity. The label column is then mapped, where 'ham' is converted to 0 and 'spam' is converted to 1, making it easier to train the model.

3. **Splitting Data into Training and Test Sets**:  
   The dataset is split into features (`X`, which contains the messages) and labels (`y`, which contains the corresponding class labels). The data is then split into training and testing sets using `train_test_split()`, with 80% of the data used for training and 20% for testing.

4. **Vectorization**:  
   **TfidfVectorizer** is used to convert the text data into numerical format that can be fed into the machine learning model. This method calculates the term frequency-inverse document frequency (TF-IDF) for each word in the messages, which reflects how important a word is within a message relative to its frequency across the entire dataset. The `fit_transform()` method is applied to the training data, while the `transform()` method is applied to the testing data.

5. **Model Training and Prediction**:  
   The **Multinomial Naive Bayes (MultinomialNB)** model is trained using the `fit()` method, where the model learns to predict the label (spam or ham) based on the TF-IDF values of the training data. After training, predictions are made on the test set using the `predict()` method.

6. **Evaluation**:  
   The performance of the model is evaluated using three main metrics:
   - **Accuracy**: This metric shows the proportion of correctly predicted labels to the total number of test samples. It is calculated using `accuracy_score()`.
   - **Classification Report**: This includes precision, recall, and F1-score for each class (ham and spam) and provides an overview of the model's performance. It is generated using `classification_report()`.
   - **Confusion Matrix**: This shows the number of true positives, false positives, true negatives, and false negatives, which helps in understanding the model’s prediction errors. The confusion matrix is visualized using **Matplotlib**.

7. **Confusion Matrix Visualization**:  
   The confusion matrix is displayed using a color-coded plot. The `matshow()` function of **Matplotlib** is used to generate the heatmap of the confusion matrix, and `ax.text()` adds the counts to each cell in the matrix. This visualization helps users easily identify how well the model is distinguishing between ham and spam messages.

 **Execution in Jupyter Notebook**

The code is implemented in a **Jupyter Notebook**, which allows for an interactive environment where each step can be executed individually. The **Jupyter Notebook** interface is ideal for running this code because it allows users to view the output of each code cell, such as the dataset preview, classification metrics, and the confusion matrix visualization, in real-time. It also provides an easy way to debug, tweak, and re-run specific parts of the code as needed, making it a perfect platform for experimenting with machine learning models.

**Conclusion**

This spam classification model demonstrates how to use **Scikit-learn** and **Pandas** for text classification tasks. It processes the text data, trains a **Naive Bayes** classifier, and evaluates the model's performance. The use of **Jupyter Notebook** facilitates the interactive and step-by-step execution of the code, making it easy to experiment with different datasets, models, and evaluation metrics.
***OUTPUT***:![Image](https://github.com/user-attachments/assets/6621063d-3d0e-48ac-b718-0c0919b88d83)
