# Web-Analysis-Project
R Project-British Conservative Voting

## 1. Object
The project consists of two main objectives. Firstly, this project is going to analyze British election panel study 
data analysis from 1997-2001 by employing several machine learning models. We aim to investigate and 
summarize the attributes of voters for Conservative Party and Non-Conservative Party. Secondly, based on the 
analysis above, we strive to figure out the reason why Conservative Party stepped down from the stage and offer 
corresponding recommendation for Conservative Party. 

## 2. Data Review  
 
### 2.1 Dataset Description 
The dataset analyzed in this project is the British election panel study data analysis from 1997-2001, which is R package dataset. 
This dataset comprises of 10 variables, with each has 1525 instances.  

#### 2.1.1 Attributes Review 
The attribute called “vote” in the dataset worked as the target attributes in this project. The target attribute consists 
of three values, “Liberal Democrat”, “Labor” and “Conservative” respectively. As this project focus on the 
Conservative Party, we divide the dependent variable to be “Conservative” and “Non-Conservative”.  

|        |Frequency  | Percentage   |
|--------|:---------:|:-------------:|
|Class 1 (Conservative)| 462| 0.3|
|Class 2 (Non-Conservative)| 1063 |0.7|

In total, there are 8 categorical variables. Except for the gender, the rest of 7 variables are ordinal  

|Qualitative Attribute  |Explanation | 
|--------|:---------:|
|Economic.cond.national|Measurement of current national economic level, from 1 to 5.|
|economic.cond.household|Assessment of current household economic conditions, 1 to 5.|
|Blair|Assessment of the Labour leader, 1 to 5.|
|Hague|Assessment of the Conservative leader, 1 to 5.|
|Kennedy|Assessment of the leader of the Liberal Democrats, 1 to 5.|
|Europe|an 11-point scale that measures respondents' attitudes toward European integration. High scores represent ‘Eurosceptic’ sentiment.|
|political.knowledge|Knowledge of parties' positions on European integration, 0 to 3.|
|gender|Gender of voter, female or male.|

#### 2.1.2 Dataset Exploration 
In this part, we first check out whether there is multicollinearity problem in this dataset using the correlation 
matrix and heatmap. It can be observed that the correlation between the variables are all smaller than 0.5, implying 
that there is no strong correlation between the variables. For the correlation between independent variables and 
the target variables “vote”, the variables “Hague”, “Blair” and “Europe” are most correlated to the target variables, 
with whose correlations are 0.47, -0.43 and 0.38 respectively. 

## 3. Data Preprocessing 
The data preprocessing part is divided into missing value detection and data transformation.  
 
### 3.1 Missing Value Detection  
Missing value detection plays an important role in data preprocessing as these values will enviably post 
significant effect on model performance. The occurrence of missing value always results from incorrect data and 
unavailable data. In R, we use sum(is.na( )) function to check missing value and  it is concluded that there is no 
missing value for our dataset. 

### 3.2 Data Transformation 
The aim of data transformation is to uniform the format of data and simplify the data mining process. In this part, 
we first use ifesle function to convert the target variable “vote” to be “Conservative” and “Non-Conservative”. 
The “Conservative” class will be the class 1 and the “Non-Conservative” be the class 0. Also, we convert the 
attribute “gender” to numerical form, with male and female belong to class 1 and class 0. 

### 3.3 Train Test Split 
To validate the model, we conduct train test split. The train data taken up to 70% of the dataset while the rest of 
30% are test data. The model will be trained using training data and then be evaluated employing test data.  

## 4. Methodology 
To analyze the dataset, we first apply 6 models to train the dataset. Then we up-sample the minority class so as 
to achieve balanced class.  
### 4.1 Model Training  
Six models are applied in this project.  They are Logistic Regression, Linear Discriminant Analysis (LDA), 
Random Forest, Decision Tree, Gradient Boosting Machine (GBM) and Support Vector Machine (SVM). The 
confusion matrix, precision, recall, F1 score and AUC will be computed for the model comparison and evaluation. 

### 4.2 Up-Sampling 
for Unbalanced Class As mentioned above, within 1525 instance of “vote”, 462 are conservative while the rest are not. With such 
unbalanced class, employing the accuracy, AUC, recall and precision ratio to measure model performance will 
be misleading. To solve this problem, we up-sample the minority class using random over sampling. The random 
over-sampling will duplicate the existing observation in the minority class continuously until the number of 
observations in minority class is equal to that of majority class. After up-sampling, the number of observations 
in class 0 and class 1 are 728. 

## 5. Data Mining 
### 5.1 Logistic Regression (LR) 
We first fit the logistic model with all variables and the result is shown below. There are total 35 variables. It can 
be observed that several variables including “economic.cond.national” and “economic.cond.household” are 
insignificant as their p-value are greater than 0.05. And we have an accuracy of 86% and 0.902 of AUC, which 
are really high. However, as we known that adding too many irrelevant variables not only decrease the adjusted 
R square, but also may lead to overfitting problem, especially for dataset that has a small sample size. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.76 |0.72| 0.91 |0.74 |0.86 | 

Therefore, we remove those insignificant variables and refit the model. The result is shown below. The accuracy 
drops to 85% and AUC drops to 0.899. Compared with the performance of complete model, the reduced model 
slightly weakens the whole model performance by drop down all the ratios we used for evaluation.  
 
|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.75| 0.72| 0.91| 0.73 |0.85  | 

Then, we up-sample the reduced model so as to deal with the unbalanced class problem. Finally, we get an 
accuracy of 80%. The number of true positive have been increase from 91 to 102, which improve the precision 
rate from 0.75 to 0.84. In the meanwhile, it sacrificed the recall ratio, F1 score and accuracy. And the AUC 
slightly drop from 0.899 to 0.894. 
 
|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.84 |0.58 |0.93| 0.69| 0.80   | 
 
### 5.2 Linear Discriminant Analysis (LDA) 
We first fit the completed model using linear discriminant analysis and the corresponding outcomes are presented 
below. Compared with that of the logistic model, linear discriminant analysis has improved the precision ratio 
from 0.76 to 0.79 and the F1 score from 0.74 to 0.76. And we get AUC of 0.899, which is slightly improved. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.79| 0.73| 0.92 |0.76| 0.86   | 

After up-sampling, similar to the logistic regression, the result of LDA after up-sampling enhance the precision 
ratio by sacrificing other ratios, especially the recall ratio which plummeted from 0.73 to 0.59. The AUC keeps 
unchanged and the threshold of ROC is lowered to 0.436.  

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.86 |0.59 |0.94 |0.70 |0.81    | 

### 5.3 Recursive Partitioning Tree (RPT) 
The idea of this algorithm is to find split condition to maximize information gains. The decision tree can be 
employed to cope with both regression and classification task. It consists of root node, internal node and leaf 
node. The root node is the start of the tree while the difference between internal node and leaf node is that internal 
node has at least one child while the latter represent the end of branch and there is no further splitting. There are 
many stop criterions for the decision tree, including maximum depth and maximum leaf nodes setting.  
Based on it, we get precision of 0.60, recall of 0.7 and F1 score of 0.65, which are greatly worse than former 
algorithm. From the plot of ROC, the AUC is also much lower than former models, only 0.789. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.60 |0.70| 0.86| 0.65 |0.83    | 

To further see how this model performs, we up sampled the data. Surprisingly, the precision increases to 0.82, 
but recall drops to 0.56 and accuracy is lower. The tree nodes also increase and become more complicated while 
AUC increases to 0.831. Overall, it performs better after up-sampling but still performs worse than other models. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.82 |0.56 |0.92 |0.66 |0.78   | 

### 5.4 Conditional Inference Tree (CIT) 
The conditional inference tree is a special kind of trees. It is recursive partitioning with binary splits and early 
stopping in a conditional inference framework. Compared with traditional decision tree, the conditional tree helps 
to overcome the overfitting, selection bias towards covariables with many possible splits, difficult interpretation 
due to selection bias and helps select variables. It has main two steps: variable selection and search for split point. 
Thus, by using this algorithm, we get an accuracy of 82% with precision of 0.65, recall of 0.68 and F1 score 0.66. 
It seems that the model performs not too good and AUC is only 0.764. The tree visualization demonstrated below 
shows that the root node is “Hague”, followed by “Blair”. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.65 |0.68 |0.87 |0.66| 0.82    | 

After up-sampling, the accuracy drops to 76% with precision of 0.84, recall of 0.53 and F1 score of 0.65 while 
AUC increases to 0.821. For the tree visualization, the tree structure is more complex than that before up
sampling. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1|0.84| 0.53| 0.93 |0.65| 0.76  | 

### 5.5 Random Forest (RF) 
The result of random forest without up-sampling presented below have a better performance than both recursive 
partitional tree and conditional inference tree in terms of precision, recall, F1 score, accuracy and AUC. The 
corresponding AUC is 0.902, much higher than that of RPT and CIT whose AUC are only 0.789 and 0.764 before 
up-sampling.

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.70 |0.73| 0.89| 0.71| 0.85    | 

After up-sampling, the performance of this model had been improved slightly, though the recall ratio and AUC 
dropped by 0.01 and 0.003 respectively. Also, we rank the feature selection based on the random forest model.  
Similar to CIT, the “Hague” is the most important variable. But what is different is that random forest tree 
“Europe” as the second most important while that for the CIT is “Blair”.   


|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.73 |0.72| 0.90| 0.73| 0.85   | 

### 5.6 Gradient Boosting Machine (GBM)  
Concerning the result below, the GBM model attained the highest precision ratio before up-sampling among all 
the models we selected for analysis, namely 0.86. While other ratios are similar to other models. What is 
interesting is that the GBM model rank the “Europe” to be the most important variables, followed by “Hague”. 
This ranking is different from the previous models we discussed. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.86 |0.63| 0.94| 0.73| 0.83   | 

After up-sampling, the precision plummeted from 0.86 to 0.70 while the recall ratio increased from 0.63 to 0.73. 
The F1 score dropped by 0.02. For the importance of the features, the ranking does not changed, but the 
importance of “Hague” have been increased from 22.66 to 24.45. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.70 |0.73| 0.89| 0.71| 0.85  | 

### 5.7 Support Vector Machine (SVM) 
From the result below, it can be seen that the recall ratio and accuracy of SVM model before up-sampling are 
0.72 and 0.85 respectively, which is higher than that of GBM model whose ratios are only 0.63 and 0.83. But 
the rest of ratios for the SVM model are lower than that of GBM model. 


|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.70 |0.72| 0.89| 0.71| 0.85    | 

After up-sampling, contrast to the GBM model, the precision ratio is enhanced after up-sampling, from 0.70 to 
0.82, which pushed down the recalled ratio from 0.72 to 0.59. Due to the dramatic reduction in the recall ratio, 
the F1 score is lowered down to 0.68. While the AUC ratio increases by a little. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1|0.82 |0.59| 0.92| 0.68| 0.80     | 

### 5.8 K Nearest Neighbors (KNN) 
Where the Y refer to the type of outcome and j indicates one of possible type. The overall result of KNN model 
are worse than that of GBM and SVM model before up-sampling. It AUC is only 0.644, which is regarded to be 
the worst among all the 9 models. 
|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.65 |0.69| 0.87| 0.67| 0.83     | 

After up-sampling, the KNN model has a poor performance with the F1 score of 0.64, though the precision have 
been increased to 0.84. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.84 |0.52| 0.92| 0.64 |0.75     | 

### 5.9 Navïe Bayes (NB) 
The algorithm of Navïe Bayes is introduced in the part of LDA. The main advantage of NB model is that it 
performs well for categorical variables and for dataset that contains small sample size. But the weakness of this 
model is that NB is not able to make a prediction for a categorical variable that does not incur in the training 
dataset but in the test dataset.  
The result of NB before up-sampling is quite similar to that of LDA, except that the precision ratio of NB is 
slightly higher than that of LDA by 0.02. The rest of the ratios are the same for these two models. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.81 |0.71| 0.93| 0.76| 0.86     | 

After up-sampling, the whole performance of NB became poorer in comparison to that before up-sampling, 
especially for the recall ratio. When compared to result of LDA model after up-sampling, the result is almost the 
same, except that the accuracy of NB is slightly lower. 

|   |Precision |Recall| Specificity| F1 score| Accuracy|
|--------|:---------:|:-------------:|:------:|:----:|:-----:|
|Class 1| 0.86| 0.59 |0.94| 0.70| 0.80      | 

## 6. Model Evaluation 
In our evaluation, we first look at the F1 score since this ratio remain unbiased even in unbalanced class and thus 
trustworthy. When a sample exist unbalanced class, it may be misleading to consider accuracy, precision and 
recall solely. Concerning the F1 score, it can be seen that the F1 score for several models like logistic regression, 
linear determinant analysis, and gradient boosting machine, dropped after up-sampling. The reason behind is that 
up-sampling have strengthened the model ability to identify the correct items and thus raise the precision ratio. 
A fortified precision lowers down recall, which further brings down the F1 score. However, as mentioned above, 
the precision, recall and accuracy will be misleading before up-sampling, we only focus on models after up
sampling. Based on the F1 score, we select four models, they are the LDA complete model after up-sampling, 
RF complete model after up-sampling, GBM complete model after up-sampling and NB complete model after 
up-sampling. Among the four models, the F1 score of RF complete model ranks the top. Then we focus on the 
recall ratio since the benefits of identifying a right voter are greater than the cost of classifying a wrong voter. 
According to the recall ratio, the GBM complete model after up-sampling earns the highest recall ratio of 0.73. 
Therefore, this model is selected as our best model for further analysis. 

## 7. Model Improvement & Recommendations 
There are two suggestions for the model improvements. Firstly, as we mentioned above, there are only 1525 
observations in this dataset which may lead to overfitting problem. Thus, our first advice is to collect expand the 
information and enlarge the observation number. Secondly, there are only 9 independent variables included in 
our dataset. There should be other relevant attributes that should be also considered in our model, such as 
education level.  
 
Moreover, there are other three recommendations for the Conservative Party. Firstly, the Conservative Party 
should solve the conflicts of politics view in their own party and define a clear ditopological line. The resignation 
of Theresa May indicates that the internal contradiction conflict in the Conservative Party still exists in the 2019. 
Secondly, to maintain its reputation, the Conservative Party should exert more effort in dealing with the resource 
allocation and tax system. Rich people should pay more tax and poor people should obtain more resources to 
improve life quality. However, paying excessive tax is also not fair to the rich people as they get their income by 
hard work. Thirdly, Conservative Party should deal with the Brexit problem more carefully. 

## 8.  Conclusion 
In conclusion, to explore the British election panel study data, we select 9 models which are logistic regression 
(LR), the linear determinant analysis (LDA), the recursive partitioning tree (RPT), the conditional inference tree 
(CIT), random forest(RF), gradient boosting machine (GBM), support vector machine (SVM) and naive bayes 
(NB). Based on the recall ratio and F1 score, we concluded that the gradient boosting machine complete model 
after up-sampling earns the best performance.  
 
To interpret our finding, we make use of the correlation matrix as well as the feature importance acquired by 
models like RPT, CIT, FR and GBM. We observed that the “Hague”, “Blair” and “Europe” are the most important 
features, meaning that the political views, the leading style and their attitude toward the Brexit highly influence 
the voters’ decision on whether they should support the Conservative Party. 
 
Finally, to improve the whole model performance, we offer two suggestions. One is to collect more observation 
so as to enlarge the sample size. The other one is to add more correlated features like the education level into the 
analysis. While for the Conservative Party, we strongly advice it to cope with internal contradiction conflict, to 
set up a define a clear ditopological line and to treat the Brexit problem cautiously. 
