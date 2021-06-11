Introduction 

K-Nearest Neighbors (KNN) classification is a simple supervised learning algorithm that utilizes similarity metrics to label an observation point based on the labels of surrounding points. KNN captures a representational space on a particular plane for a set of labels, without implementing traditional generalization techniques as other predictive algorithms perform. This case study examines the effectiveness of KNN on predicting the popularity of a given article published by Mashable. 

The dataset in this examination contains 39,797 records with 58 predictive attributes and 1 target attribute (number of shares, which can be used as a measure of popularity).

Data Visualization and Formatting 

To begin with, an autocorrelation matrix was created to see if there is a linear relation to any variables. This can be seen in Figure 1. We can see a relation between data channels. It makes sense that if an article is part of one data channel it won’t be a part of another. There is also a correlation between the LDA and the day of week.  Overall, we can see that many of these variables are highly correlated due to a variety of lurking variables, so a dimensionality reduction technique would need to be implemented (such as Principal Component Analysis (PCA)) in order to reduce the sensitivity of the K-Nearest Neighbors model (KNN). 

Figure 2 shows a violin plot of the dependent variable (the amount of shares). It can be seen that the data is relatively normally distributed but with many outliers. The relative normal distribution can be seen in figure 3, though it can be seen to be skewed a bit to the left.  

We decided to split the data into bins based on popularity (by shares). In a research paper done by K. Fernandez, et al (Fernandez et al., 2015), a binary classification model was created where 1400 shares was set as the threshold for popularity (below 1400 shares means the article is unpopular, above 1400 shares means the article is popular). We decided to add another bin to hold the outliers, which can be listed as any value above 5500 (anything above the upper fence on the violin plot). These values can be considered to be extremely popular. 

Figure 4 shows the value counts for each bin. It can be seen that there are much less articles in the extremely popular category than there are in the popular and unpopular. Unfortunately when we’re creating our final models, this imbalance in data might lead to overfitting in the final model. 

Initial Insights 

Figure 5 shows the mean values of the different independent features given the popularity bins. This graph would be very valuable for a manager at Mashable, as they can see what features would make an article more likely to be more or less popular. It can be seen that keywords have a massive impact on an article being extremely popular. The article topics and their news channel also have a big impact on their likelihood to become popular. The number of links increasing also increases the chance of an article to go viral. The day of the week can be seen to have an impact on popularity as well. 


Implementation of KNN

A Principal component Analysis was first implemented to reduce dimensionality. Figure 6 shows a graph of the cumulative explained variance of the PCA analysis based on the number of components. It was found that 36 components explained 95.5% of the explained variance in the data, so the data was then reduced down to 36 components. 

The data was then fit into a K-Nearest Neighbors model. Figure 7 shows the model accuracy for each of the models, testing between 0 and 50 neighbors. We want to minimize the amount of neighbors while maximizing the accuracy for the optimal model. It was found that 11 neighbors leads to a 55% accuracy. 

Overfitting and Underfitting

After implementing the KNN model, bagging and gradient boosting classifiers were used as an ensemble classifier to enhance the performance of the model. This will decrease either overfitting or underfitting of the original model. Essentially, assembling the KNN model with a bagging classifier will reduce the complexity of the model, in order word, reducing overfitting. Adding gradient boost will do the exact opposite of bagging. It will increase the complexity of the model, hence, reducing underfitting. We used KNN with 24 neighbors as the base case which has 55.96% accuracy. Adding a bagging classifier will reduce the accuracy down to 54.21% while adding an XGBoosting classifier will increase the accuracy to 56.78%. This means that our KNN model has a slight underfitting, and we can therefore have a better performance by adding gradient boosting. Since we did an PCA in the previous process, the model is most likely not overfitted. Therefore, we also tried taking out the PCA step and running the model ensemble with bagging alone to reduce dimension. This only gives us a 55.15% accuracy which is lower than the base case, while significantly increases the computational expense.

After adding either bagging or gradient boosting to the KNN model, we naturally think about the random forest classifier which is essentially a stronger bagging classifier. We then experience this model to see how it performs against KNN.

Random Forest

Random forest classifier yields a better accuracy than the KNN model since it is a better choice for high dimensional data. We can eliminate the need for the PCA step while using random forest classifiers because the model will automatically choose the important variables. The only benefit of using PCA with random forest is to reduce computational expense, which is not a problem here since the data is small. The resulting accuracy of the random forest classifier is 58.42% which is better than the KNN model even after adding XGBoosting classifier to it. We then take one more step forward by joining a random forest classifier and XGBoosting classifier together using a voting classifier. On soft voting policy, the model gives 60.67% accuracy which is almost 5% more accurate than the KNN model. However, The drawback of random forest classifiers is also significant, it does not give the business user much insights about the importances of each individual data attribute. In this case, using the random forest classifier does not meet the business goal of finding important attributes. However, if the business goal of this project is aiming at a model of the highest accuracy. The random forest classifier is a better choice than the KNN model. 

Discussion

An important decision in terms of model implementation is how to best utilize the collected data to achieve the end goal. In certain contexts, more information is not always better. For example, the model presented in this case utilized principle component analysis to reduce the dimensionality of the data. One limitation of KNN is that it does not perform well with high dimensional data as calculating distance from one point to another in a high dimensional space does not map well to similarity. However, the implementation of KNN implies that the more observations collected, the better the algorithm will perform. From a management perspective, this justifies the cost of collecting, processing and storing large-scale data. Furthermore, initial decisions should both leverage domain knowledge and information collected from exploratory data analysis. The discretization of the target variable directly influences the performance of the model. Prior to grouping outliers, the model was running an accuracy of 50%, this 5% increase in accuracy is attributed to a human-made decision prior to implementing KNN. It is important for executive level management to understand the underlying decisions made that impact the model.

From the perspective of executive leadership, the biggest advantage KNN provides is it’s transparency in implementation. Unlike other algorithms, KNN is conceptually straightforward, furthermore a label is not extracted from a generalized model. Consequently, it is easier to extract a set of points that is similar to a given record and explain these similarities based on the defined dimensions. The transparency allows for better decision making, for instance outliers are easily spotted. Additionally, a publisher not only has the ability to predict how popular an article will be, but can see how similar articles performed. This information can be leveraged to better improve the popularity of future articles by either investing in “weaker” attributes to increase their popularity, or investing in articles that are likely to be popular based on historical data. 

Where KNN falls short is its ability to select strong features for the predictive model. The task in this case essentially predicts consumer behavior. This is a difficult analysis as a consumer’s decision making process is vastly complex. KNN essentially leverages one feature: distance as a model of similarity. While this implementation has its advantages as previously discussed, it does not produce a high accuracy. The documentation from the dataset was only able to get a maximum accuracy of 67%/ utilizing different models and just 2 labels. Future work should focus on stronger feature selection models. For example, neural networks have been increasing in popularity, with some publications centering on utilizing Convolutional Neural Network in tandem with KNN (Gallego et al., 2020, p. 99321). Overall, managers need to make a variety of decisions in order to leverage the information data mining provides. Utilizing both domain and knowledge of advantages and shortcoming of algorithms can build a strong predictive model. These models can lead to better decisions and resource allocations to maximize return on investment.


Works Cited

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

Gallego, A. J., Calvo-Zaragoza, J., & Rico-Juan, J. R. (2020). Insights Into Efficient k-Nearest Neighbor Classification With Convolutional Neural Codes. IEEE Access, 8, 99312–99326. https://doi.org/10.1109/access.2020.2997387

Appendix: 

Figure 1: Autocorrelation heatmap of all variables 



Figure 2: Violin distribution of the dependent variable (shares). 


Figure 3: A histogram of the shares with the outliers removed




Figure 4: Bar plot of the value counts based on the different bins


Figure 5: Means of different independent variables for each popularity bin




Figure 6: PCA graph shows that 36 components explain 95.5% of the variance. 


Figure 7: Accuracy Score for K-Nearest Neighbor models
