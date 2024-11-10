library(caret)
library(e1071)      # For Naive Bayes
library(class)      # For KNN
library(rpart)      # For Decision Tree
library(cluster)    # For K-Means
library(MLmetrics)  # For additional metrics
library(ggplot2)    # For plotting
library(dplyr)
library(tidyr)

# Load data
data <- read.csv(file.choose())

# Data preprocessing
data$Genre <- as.factor(data$Genre)  # Convert 'Genre' to a factor
data$SpendingScore <- as.factor(ifelse(data$SpendingScore < 33, "Low", 
                                       ifelse(data$SpendingScore < 67, "Medium", "High")))  # Categorize Spending Score

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$SpendingScore, p = 0.7, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Define a function to calculate evaluation metrics
calculate_metrics <- function(predictions, actual) {
  accuracy <- Accuracy(predictions, actual)
  precision <- Precision(predictions, actual, positive = "High")
  recall <- Recall(predictions, actual, positive = "High")
  f1 <- F1_Score(predictions, actual, positive = "High")
  error_rate <- 1 - accuracy
  return(data.frame(accuracy, precision, recall, f1, error_rate))
}

# KNN Model
knn_model <- knn(train = trainData[, c("Age", "AnnualIncome")],
                 test = testData[, c("Age", "AnnualIncome")],
                 cl = trainData$SpendingScore, k = 5)

# Ensure the predicted and actual labels have the same factor levels
knn_model <- factor(knn_model, levels = levels(testData$SpendingScore))

# KNN Evaluation
calculate_metrics(knn_model, testData$SpendingScore)
knn_metrics <- calculate_metrics(knn_model, testData$SpendingScore)

# Reshape the metrics for plotting
knn_metrics_long <- gather(knn_metrics, key = "variable", value = "value")

# Plotting KNN model's evaluation metrics
ggplot(knn_metrics_long, aes(x = variable, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("KNN Model Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ------------------------

# Naive Bayes Model
nb_model <- naiveBayes(SpendingScore ~ Age + AnnualIncome, data = trainData)
nb_predictions <- predict(nb_model, testData)

# Naive Bayes Evaluation
calculate_metrics(nb_predictions, testData$SpendingScore)
nb_metrics <- calculate_metrics(nb_predictions, testData$SpendingScore)

# Reshape the metrics for plotting
nb_metrics_long <- gather(nb_metrics, key = "variable", value = "value")

# Plotting Naive Bayes model's evaluation metrics
ggplot(nb_metrics_long, aes(x = variable, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Naive Bayes Model Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------

# Decision Tree Model
dt_model <- rpart(SpendingScore ~ Age + AnnualIncome, data = trainData, method = "class")
dt_predictions <- predict(dt_model, testData, type = "class")

# Decision Tree Evaluation
calculate_metrics(dt_predictions, testData$SpendingScore)
dt_metrics <- calculate_metrics(dt_predictions, testData$SpendingScore)

# Reshape the metrics for plotting
nb_metrics_long <- gather(nb_metrics, key = "variable", value = "value")

# Plotting Naive Bayes model's evaluation metrics
ggplot(nb_metrics_long, aes(x = variable, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Naive Bayes Model Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------

# Logistic Regression Model
trainData$Spending.Binary <- ifelse(trainData$SpendingScore == "High", 1, 0)
testData$Spending.Binary <- ifelse(testData$SpendingScore == "High", 1, 0)

log_reg <- glm(Spending.Binary ~ Age + AnnualIncome, data = trainData, family = binomial)

# Logistic Regression Predictions
log_reg_pred <- predict(log_reg, newdata = testData, type = "response")
log_reg_class <- ifelse(log_reg_pred > 0.5, 1, 0)

# Logistic Regression Evaluation
calculate_metrics(as.factor(log_reg_class), as.factor(testData$Spending.Binary))
log_reg_metrics <- calculate_metrics(as.factor(log_reg_class), as.factor(testData$Spending.Binary))

log_reg_metrics_long <- gather(log_reg_metrics, key = "variable", value = "value")

# Plotting Logistic Regression model's evaluation metrics
ggplot(log_reg_metrics_long, aes(x = variable, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Logistic Regression Model Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# -----------------------

# K-Means clustering (for clustering purposes, not supervised)
set.seed(123)
kmeans_model <- kmeans(trainData[, c("Age", "AnnualIncome")], centers = 3)

# Add the predicted clusters as a new column in trainData
trainData$Cluster <- factor(kmeans_model$cluster)

# Evaluate K-Means (using Accuracy, Precision, Recall, F1 as a rough evaluation based on clusters)
# For simplicity, use a common label for comparison: assume 'High' spending is the highest spending category
trainData$Cluster <- factor(ifelse(trainData$Cluster == 1, "Low", ifelse(trainData$Cluster == 2, "Medium", "High")))

# K-Means Evaluation
calculate_metrics(trainData$Cluster, trainData$SpendingScore)
kmeans_metrics <- calculate_metrics(trainData$Cluster, trainData$SpendingScore)

# Reshape the metrics for plotting
kmeans_metrics_long <- gather(kmeans_metrics, key = "variable", value = "value")

# Plotting K-Means model's evaluation metrics
ggplot(kmeans_metrics_long, aes(x = variable, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("K-Means Clustering Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Combine all metrics
all_metrics <- rbind(
  data.frame(Model = "KNN", knn_metrics),
  data.frame(Model = "Naive Bayes", nb_metrics),
  data.frame(Model = "Decision Tree", dt_metrics),
  data.frame(Model = "Logistic Regression", log_reg_metrics),
  data.frame(Model = "K-Means", kmeans_metrics)
)

# Reshape metrics for plotting
all_metrics_long <- gather(all_metrics, key = "metric", value = "value", -Model)

# Plotting individual metrics for each model
ggplot(all_metrics_long, aes(x = Model, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~ metric, scales = "free") +
  theme_minimal() +
  ggtitle("Model Comparison Across Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
