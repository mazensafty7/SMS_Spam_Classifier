# SMS Spam Classifier using Naive Bayes

![1_Fm58r_RQ53sEHfwFa28LpA](https://github.com/user-attachments/assets/bb8fb059-22e7-4090-a782-59f4faf2b1dc)

## Project Goal
Spam detection is one of the major applications of Machine Learning today. Most major email providers have built-in spam detection systems that classify junk mail automatically. In this project, we aim to build a model that can classify SMS messages as either spam or not spam using the **Naive Bayes algorithm**.

Spam messages often contain trigger words like "free," "win," "cash," "prize," and are usually formatted with capital letters or excessive punctuation (e.g., exclamation marks) to catch the recipient's attention. The goal of this project is to teach a machine learning model to identify these patterns and distinguish spam from non-spam messages.

The problem is a **binary classification** task, where messages are labeled as either 'Spam' or 'Ham' (not spam), and it's also a **supervised learning** task, as we train the model on labeled data.

---

## Introduction to Naive Bayes
**Bayes' Theorem** is a probabilistic algorithm that dates back to the work of Reverend Thomas Bayes. It is particularly effective in use cases like text classification, where the goal is to calculate the likelihood of an event (a message being spam) based on the occurrence of certain related features (the appearance of specific words).

Naive Bayes assumes that these features are independent, which simplifies calculations but may not always reflect real-world interactions between features. Despite this assumption, Naive Bayes performs extremely well in many text-based applications like spam detection.

To illustrate, imagine you're assessing threats based on multiple factors such as age, nervousness, and whether someone is carrying a bag. Bayes' Theorem would help you compute the overall threat level by weighing each factor independently, even though in reality, some of these factors might be correlated (e.g., a child's nervousness is less alarming than an adult's).

In the case of our spam classifier, the algorithm calculates the probability of a message being spam based on the appearance of certain keywords, independent of other features.

---

## Dataset Overview
The dataset used in this project contains labeled SMS messages marked as either 'spam' or 'ham'. Each message is preprocessed and mapped, with 'spam' being assigned a value of 1, and 'ham' a value of 0.

---

## Project Workflow

1. **Data Loading & Understanding**:
   - We import the dataset and take a closer look at the structure of the messages, understanding the proportions of spam vs. ham in the dataset.
   
2. **Data Preprocessing**:
   - Convert text labels to numeric values (1 for spam, 0 for ham).
     
3. **Splitting the Dataset**:
   - We split the data into training and testing sets to evaluate the model's performance.

4. **Text Vectorization**:
   - Apply **Bag of Words** model using `CountVectorizer` to transform the text data into numerical features representing word counts, allowing Naive Bayes to process the data.

5. **Naive Bayes Implementation Using Scikit-Learn**:
   - We implemented the **Multinomial Naive Bayes** model using Scikit-Learn's `MultinomialNB()` function.
   - Trained the model on the training data, allowing it to learn from labeled examples.

6. **Model Evaluation**:
   - After training, the model was evaluated on the test set using key performance metrics:
     - **Accuracy Score**: 0.9885
     - **Precision Score**: 0.9721
     - **Recall Score**: 0.9405
     - **F1 Score**: 0.9560

   Among these metrics, the **precision score** is particularly important for spam detection, as it represents how many of the predicted spam messages were actually spam. Our model achieved a high precision score, indicating that it is very effective at minimizing false positives (incorrectly labeling ham as spam).

---

## Example Testing
After evaluating the model, we tested it with new examples to ensure it could accurately predict whether an SMS message is spam or not. The model successfully classified test cases, reinforcing the model’s reliability.

---

## Conclusion
This project successfully demonstrates how the **Naive Bayes algorithm** can be applied to build an accurate SMS spam classifier. With an accuracy of 98.85% and a strong precision score, the model is effective at identifying spam messages while minimizing false positives. Future improvements could involve experimenting with advanced text processing techniques or trying other machine learning algorithms to further boost performance.
