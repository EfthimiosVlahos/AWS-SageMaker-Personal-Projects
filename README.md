# AWS Personal Projects

This repository showcases my collection of diverse projects that leverage AWS SageMaker and various other AWS services. The projects encompass health insurance cost prediction, forecasting weekly retail store sales, cardiovascular disease detection, and traffic sign recognition using cutting-edge deep learning techniques. Below is a concise summary of each project:

# Table of contents
- [01. Health Insurance Cost Prediction](#s1)
  - [01.1 SageMaker Linear Learner Model Scores](#s11)
  - [01.2 Artificial Neural Networks (ANN) Model Scores:](#s12)
  - [01.3 Conclusion](#s13)
- [02. Forecasting Weekly Retail Store Sales](#s2)
  - [02.1 Local XGBoost Model Results](#s21)
  - [02.2 AWS SageMaker XGBoost Model Results](#s22)
  - [02.3 Conclusion ](#s23)
- [03. Cardiovascular Disease Detection](#s3)
    - [03.1 Local XGBoost Model Evaluation](#s31)
    - [03.2 Hyperparameter Tuning with GridSearch](#s32)
    - [03.3 Dimensionality Reduction using PCA (AWS SageMaker)](#s33)
    - [03.4 XGBoost Model Training and Evaluation after Dimensionality Reduction (AWS SageMaker)](#s34)
    - [03.5 Model Deployment and Testing](#s35)
    - [03.6 Concluion](#s36)
- [04. Traffic Sign Recognition - Deep Learning Multiclassifier](#s4)
  - [04.1 Training the CNN LeNet Model using Amazon SageMaker](#s41)
  - [04.2 Conclusion](#s42)
    
# Health Insurance Cost Prediction <a name="s1"></a>

In this project, I applied two different predictive models, SageMaker Linear Learner and Artificial Neural Networks (ANN), to estimate the health insurance cost incurred by individuals based on various factors. The key findings from our model evaluations are as follows:

## SageMaker Linear Learner Model Scores: <a name="s11"></a>

- Mean Squared Error (MSE): 0.2992234559816735
- Absolute Loss: 0.30221866892877025
- Root Mean Squared Error (RMSE): 0.5470132137176154
- R-squared (R2): 0.7007765460174951
- Mean Absolute Error (MAE): 0.30221866


## Artificial Neural Networks (ANN) Model Scores: <a name="s12"></a>

- Root Mean Squared Error (RMSE): 5115.582
- Mean Squared Error (MSE): 26169176.0
- Mean Absolute Error (MAE): 2986.2544
- R-squared (R2): 0.8314370960526327
- Adjusted R-squared: 0.8262305198689303



## Conclusion <a name="s13"></a>
The SageMaker Linear Learner model demonstrates respectable performance, with a moderate R-squared value of 0.70, indicating that it explains approximately 70% of the variance in health insurance costs. The low values of MSE, RMSE, and MAE further validate the model's ability to make accurate predictions. 
The ANN model exhibits superior performance compared to the Linear Learner, with a higher R-squared value of 0.83, indicating that it explains approximately 83% of the variance in health insurance costs. The low values of RMSE, MSE, and MAE underscore the model's ability to make accurate predictions.

Overall, both models showcase their strengths in predicting health insurance costs, with the ANN model outperforming the Linear Learner model in terms of predictive accuracy. The ANN model's ability to capture complex relationships within the data contributes to its improved performance.
# Forecasting Weekly Retail Store Sales <a name="s2"></a>

In this data science case study, I focused on forecasting weekly retail store sales for specific departments using the XGBoost algorithm. The project comprises several key phases, including data collection, organization, exploratory data analysis (EDA), hypothesis testing, and model building. I leveraged both local and AWS SageMaker environments to develop and fine-tune the XGBoost model, leading to valuable insights and accurate predictions.

## Local XGBoost Model Results: <a name="s21"></a>

After performing the XGBoost algorithm locally and evaluating its performance on the test set, I obtained the following metrics:

- Root Mean Squared Error (RMSE): 9965.678
- Mean Squared Error (MSE): 99314740.0
- Mean Absolute Error (MAE): 6441.0
- R-squared (R2): 0.8086990963964283
- Adjusted R-squared: 0.80786045715448

## AWS SageMaker XGBoost Model Results: <a name="s22"></a>

I extended our analysis by utilizing AWS SageMaker to perform hyperparameter tuning, optimizing the XGBoost model's performance. The scores obtained on the test set for the SageMaker XGBoost model are as follows:

- Root Mean Squared Error (RMSE): 4202.395
- Mean Squared Error (MSE): 17660124.0
- Mean Absolute Error (MAE): 1824.0424
- R-squared (R2): 0.9659829172814032
- Adjusted R-squared: 0.9658337906441159

## Conclusion <a name="s23"></a>
The local XGBoost model demonstrated promising results with an R-squared value of approximately 0.81, indicating that it explains around 81% of the variance in the weekly retail store sales. The Adjusted R-squared value further validates the model's ability to capture meaningful relationships in the data.
I extended our analysis by utilizing AWS SageMaker to perform hyperparameter tuning, optimizing the XGBoost model's performance.
The AWS SageMaker XGBoost model significantly outperformed the local model, achieving an impressive R-squared value of approximately 0.97, indicating that it explains approximately 97% of the variance in the weekly retail store sales. The hyperparameter tuning job led to improved predictive accuracy and enhanced model generalization.



# Cardiovascular Disease Detection <a name="s3"></a>

In this Jupyter Notebook, we addressed the important task of detecting the presence or absence of cardiovascular disease in individuals based on various features related to their health and lifestyle. The dataset encompassed key attributes such as age, height, weight, gender, blood pressure readings, cholesterol level, glucose level, smoking habits, alcohol intake, physical activity, and the binary label indicating the presence or absence of cardiovascular disease.

## Local XGBoost Model Evaluation: <a name="s31"></a>

Initially, I trained and tested an XGBoost model in local mode, and the performance on the testing set was as follows:

- Precision: 0.7346628101689755
- Recall: 0.6903161207266485
- Accuracy: 0.7208571428571429

## Hyperparameter Tuning with GridSearch: <a name="s32"></a>

To further improve the model's performance, I performed grid search to optimize the hyperparameters. The tuned XGBoost model yielded enhanced evaluation scores:

- Precision: 0.7720334810438207
- Recall: 0.6728651122872265
- Accuracy: 0.7374285714285714

## Dimensionality Reduction using PCA (AWS SageMaker): <a name="s33"></a>

I explored dimensionality reduction techniques to streamline the feature space and improve model efficiency. Specifically, I employed Principal Component Analysis (PCA) using AWS SageMaker.

## XGBoost Model Training and Evaluation after Dimensionality Reduction (AWS SageMaker): <a name="s34"></a>

The XGBoost model was trained and evaluated on the data after applying PCA for dimensionality reduction. The results indicate favorable evaluation scores.

## Model Deployment and Testing: <a name="s35"></a>

Finally, I deployed the trained XGBoost model to make predictions on new data. The model exhibited favorable accuracy, precision, and recall, making it a valuable tool for detecting cardiovascular disease in individuals.

## Concluion <a name="s36"></a>
The developed XGBoost model, combined with dimensionality reduction using PCA, showcases the potential of machine learning in healthcare for accurate disease detection. Early identification of cardiovascular disease can lead to timely interventions and personalized treatment plans, positively impacting patient outcomes and reducing healthcare costs.
# Traffic Sign Recognition - Deep Learning Multiclassifier <a name="s4"></a>

The primary objective of this project was to develop a powerful traffic sign recognition system by harnessing the capabilities of TensorFlow and Keras and implementing the LeNet architecture. The application of deep learning in this context has immense implications for enhancing road safety, facilitating autonomous vehicles, and enabling intelligent traffic management. The project showcases the potential of cutting-edge deep learning techniques in addressing real-world challenges and paving the way for safer and more efficient transportation systems.

## Training the CNN LeNet Model using Amazon SageMaker <a name="s41"></a>

In pursuit of this goal, I employed Amazon SageMaker, a comprehensive cloud-based machine learning platform, to train the LeNet model at scale. SageMaker's capabilities enabled me to efficiently train the deep neural network on a large dataset, leveraging the elasticity and computational resources of the cloud. This significantly reduced the training time and facilitated the optimization of the model's performance.

Creating a train-cnn.py file allowed me to construct the LeNet model framework, and I used SageMaker to train the CNN on the traffic sign dataset. The end result metrics showcased the model's effectiveness, with a validation loss of 0.7172 and a validation accuracy of 0.8400. These metrics indicate the model's ability to generalize well to unseen data, a crucial aspect for real-world applications.
## Conclusion <a name="s42"></a>

The successful development and training of the LeNet model for traffic sign recognition illustrate the potential of deep learning techniques in revolutionizing transportation systems. The accuracy achieved by the model on the validation set demonstrates its capability to identify and classify various traffic signs with high precision, which can be vital in critical scenarios.

In conclusion, these projects demonstrate the power of advanced machine learning and deep learning techniques in solving real-world problems and enhancing decision-making processes. The results obtained from these projects have significant implications for healthcare, retail, and transportation industries. By leveraging cutting-edge technologies and cloud-based machine learning platforms, I have showcased the potential for scalable and accurate predictive modeling, setting the stage for future advancements in data science and machine learning.
