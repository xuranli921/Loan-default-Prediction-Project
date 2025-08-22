## Import the cleaned data
data <-read.csv("train_new.csv")


## load necessary packages
library(dplyr)
library(class)
library(ROCR)
library(caret)
library(Metrics)
# Delete the row X
data <- data[, -which(names(data) == "X")]
set.seed(123)  # For reproducibility

# Calculate the number of observations to include in the test set
test_size <- ceiling(0.25 * nrow(data))

# Randomly select indices for the test set
test_indices <- sample(1:nrow(data), test_size)

# Create test and training sets
test_set <- data[test_indices, ]
train_set <- data[-test_indices, ]

train_set$isDefault <- as.factor(train_set$isDefault)
test_set$isDefault <- as.factor(test_set$isDefault)


# Count the number of instances for each class in 'isDefault'
table(train_set$isDefault)

# Calculate proportions for each class
prop.table(table(train_set$isDefault))





## Make a balanced dataframe


# Function to create a balanced dataset by undersampling
create_balanced_dataset <- function(data, seed) {
  set.seed(seed)
  class_0 <- filter(data, isDefault == 0)
  class_1 <- filter(data, isDefault == 1)
  
  class_0_sampled <- sample_n(class_0, size = nrow(class_1))
  balanced_train <- bind_rows(class_0_sampled, class_1)
  
  return(balanced_train[sample(nrow(balanced_train)),])  # Shuffle rows
}

# Create multiple datasets
balanced_train1 <- create_balanced_dataset(train_set, seed = 101)
balanced_train2 <- create_balanced_dataset(train_set, seed = 102)
balanced_train3 <- create_balanced_dataset(train_set, seed = 103)


## knn model when k=5
# Train the model
knn_model <- knn(train=balanced_train1[,-23], test=test_set[,-23],
               cl=balanced_train1$isDefault,k=5)
knnmodel<-knn(train=balanced_train1[,-23], test=test_set[,-23],
              cl=balanced_train1$isDefault,k=5,prob = TRUE)
test_predictions <- knn_model

# Predict probabilities
probs<- attr(knnmodel, "prob")

# Calculate confusion matrix and performance metrics
confusion_matrix <- table(test_set$isDefault, test_predictions)
precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
f1_score <- 2 * precision * recall / (precision + recall)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Display the metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

## Calculate AUC
auc(test_set$isDefault,test_predictions)

## Calculate confusion matrix
actual <- test_set$isDefault
cm <- table(actual,test_predictions)
cm

## plot precision @k

test_1<-data.frame(probs=as.vector(probs),isDefault=as.numeric(test_set$isDefault)-1)
plot_1<-test_1%>%
  arrange(desc(probs))%>%
  mutate(knn_precision=cumsum(test_1$isDefault)/1:n())

p_1<-ggplot()+
  geom_line(data=plot_1,
            aes(x=1:nrow(plot_1),y=knn_precision))+
  labs(x="Top K cases",y="Precision",title="Precision-at-k in KNN")+
  theme_minimal()
p_1
