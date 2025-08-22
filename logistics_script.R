# data cleaning
library(tidyverse)
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# transform string variable "grade" and "subGrade" into factors:
train$grade <- factor(train$grade, 
                      level = c("A", "B", "C", "D", "E", "F", "G"))

train$subGrade <- factor(train$subGrade)

test$grade <- factor(test$grade, 
                      level = c("A", "B", "C", "D", "E", "F", "G"))

test$subGrade <- factor(test$subGrade)

# transform "employmentLength" to numeric values:
train$employmentLength <- as.numeric(gsub("\\D+", "", train$employmentLength))
test$employmentLength <- as.numeric(gsub("\\D+", "", test$employmentLength))

# transform character variable "issueDate" to date varaible, don't know ear
train$issueDate <- as.Date(train$issueDate, format = "%Y/%m/%d")
test$issueDate <- as.Date(test$issueDate, format = "%Y/%m/%d")

train$earliesCreditLine <- as.Date(paste0(train$earliesCreditLine, "-01"), 
                                     format = "%b-%y-%d")
test$earliesCreditLine <- as.Date(paste0(test$earliesCreditLine, "-01"), 
                                     format = "%b-%y-%d")

# check NA
missing_values_train <- apply(train, 2, function(x) sum(is.na(x)))
print(missing_values_train)

missing_values_test <- apply(test, 2, function(x) sum(is.na(x)))
print(missing_values_test)
 # we can see employmentLength/verificationStatus/postCode/dti/revolUtil
# title/n0-n14 have missing values

# use decision tree to fill out NA in train data
library(rpart)
tree_model_train <- rpart(employmentLength ~ loanAmnt + grade + interestRate + 
annualIncome + homeOwnership + term +regionCode, 
                    data = train[!is.na(train$employmentLength),], method = "anova")

predicted_values <- predict(tree_model_train, 
                            newdata = train[is.na(train$employmentLength),])

train$employmentLength[is.na(train$employmentLength)] <- as.integer(predicted_values)
train$dti[is.na(train$dti)] <- median(train$dti)
train$revolUtil[is.na(train$revolUtil)] <- median(train$revolUtil)
train_new <- train %>% 
  dplyr::select(-n0, -n1, -n2, -n3, -n4, -n5, -n6, -n7, -n8, -n9, -n10, -n11, 
         -n12, -n13, -n14)
train_new <- na.omit(train_new)

# use decision tree to fill out NA in test data
tree_model_test <- rpart(employmentLength ~ loanAmnt + grade + interestRate + 
                            annualIncome + homeOwnership + term +regionCode, 
                          data = test[!is.na(test$employmentLength),], method = "anova")

predicted_values_test <- predict(tree_model_test, 
                            newdata = test[is.na(test$employmentLength),])

test$employmentLength[is.na(test$employmentLength)] <- as.integer(predicted_values_test)
test$dti[is.na(test$dti)] <- median(test$dti)
test$revolUtil[is.na(test$revolUtil)] <- median(test$revolUtil)
test_new <- test %>% 
  dplyr::select(-n0, -n1, -n2, -n3, -n4, -n5, -n6, -n7, -n8, -n9, -n10, -n11, 
                -n12, -n13, -n14)
test_new <- na.omit(test_new)

# 直接移除n0-n14? 因为我们也不知道这些是什么意思，而且还有那么多缺失值

# feature selection: using random forest model
library(ranger)
library(caret)
train$isDefault <- as.factor(train$isDefault)
train_new <- train_new %>% 
  dplyr::select(-id)
rf_model <- ranger(isDefault ~ ., data = train_new, importance = "impurity")
importance <- importance(rf_model)
print(importance)
plot_data <- as.data.frame(importance) %>% 
  rownames_to_column("features")

ggplot(plot_data, aes(x = reorder(features, -importance)
                      , y = importance)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = 1000, linetype = "dashed", color = "red")+
  labs(x = "features", y = "Importance") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
# logistic regression modeling
# only use "important" features as the dataset
train_model <- train_new %>% 
  dplyr::select(-pubRec, -initialListStatus, -pubRecBankruptcies, 
         -applicationType, -policyCode)

test_model <- test_new %>% 
  dplyr::select(-pubRec, -initialListStatus, -pubRecBankruptcies, 
                -applicationType, -policyCode, -id)
# balance default and non-default data

n <- nrow(train_new[train_new$isDefault==1,])

# AUC ##########################################
# 1
sample_1 <- train_model %>% 
  filter(isDefault == 0) %>% 
  sample_n(size = n) 
sample_1 <- sample_1 %>% 
  rbind(train_model[train_model$isDefault==1,])
index_1 <- sample(1:nrow(sample_1), size = nrow(sample_1)*0.8, replace = F)
train_1 <- sample_1[index_1, ]
test_1 <- sample_1[-index_1, ]

logistic_1 <- glm(data = train_1, isDefault ~.,
                  family = "binomial")
predicted_values_1 <- as.vector(predict(logistic_1, newdata = test_1, 
                              type = "response"))
predicted_obj_1 <- prediction(predicted_values_1, test_1$isDefault)
auc_1 <- performance(predicted_obj_1, "auc")@y.values[[1]]

# 2
sample_2 <- train_model %>% 
  filter(isDefault == 0) %>% 
  sample_n(size = n) 
sample_2 <- sample_2 %>% 
  rbind(train_model[train_model$isDefault==1,])
index_2 <- sample(1:nrow(sample_2), size = nrow(sample_2)*0.8, replace = F)
train_2 <- sample_2[index_2, ]
test_2 <- sample_2[-index_2, ]

logistic_2 <- glm(data = train_2, isDefault ~.,
                  family = "binomial")
predicted_values_2 <- as.vector(predict(logistic_2, newdata = test_2, 
                                        type = "response"))
predicted_obj_2 <- prediction(predicted_values_2, test_2$isDefault)
auc_2 <- performance(predicted_obj_2, "auc")@y.values[[1]]

# 3
sample_3 <- train_model %>% 
  filter(isDefault == 0) %>% 
  sample_n(size = n) 
sample_3 <- sample_3 %>% 
  rbind(train_model[train_model$isDefault==1,])
index_3 <- sample(1:nrow(sample_3), size = nrow(sample_3)*0.8, replace = F)
train_3 <- sample_3[index_3, ]
test_3 <- sample_3[-index_3, ]

logistic_3 <- glm(data = train_3, isDefault ~.,
                  family = "binomial")
predicted_values_3 <- as.vector(predict(logistic_3, newdata = test_3, 
                                        type = "response"))
predicted_obj_3 <- prediction(predicted_values_3, test_3$isDefault)
auc_3 <- performance(predicted_obj_3, "auc")@y.values[[1]]

# 4
sample_4 <- train_model %>% 
  filter(isDefault == 0) %>% 
  sample_n(size = n) 
sample_4 <- sample_4 %>% 
  rbind(train_model[train_model$isDefault==1,])
index_4 <- sample(1:nrow(sample_4), size = nrow(sample_4)*0.8, replace = F)
train_4 <- sample_4[index_4, ]
test_4 <- sample_4[-index_4, ]

logistic_4 <- glm(data = train_4, isDefault ~.,
                  family = "binomial")
predicted_values_4 <- as.vector(predict(logistic_4, newdata = test_4, 
                                        type = "response"))
predicted_obj_4 <- prediction(predicted_values_4, test_4$isDefault)
auc_4 <- performance(predicted_obj_4, "auc")@y.values[[1]]

# 5
sample_5 <- train_model %>% 
  filter(isDefault == 0) %>% 
  sample_n(size = n) 
sample_5 <- sample_5 %>% 
  rbind(train_model[train_model$isDefault==1,])
index_5 <- sample(1:nrow(sample_5), size = nrow(sample_5)*0.8, replace = F)
train_5 <- sample_5[index_5, ]
test_5 <- sample_5[-index_5, ]

logistic_5 <- glm(data = train_5, isDefault ~.,
                  family = "binomial")
predicted_values_5 <- as.vector(predict(logistic_5, newdata = test_5, 
                                        type = "response"))
predicted_obj_5 <- prediction(predicted_values_5, test_5$isDefault)
auc_5 <- performance(predicted_obj_5, "auc")@y.values[[1]]

mean_auc <- (auc_1 + auc_2 + auc_3 + auc_4 + auc_5)/5
print(mean_auc)

# Precision-at-k% ##########################################
# 1
# Order the predictions by probability of being class 1
ordered_indices_1 <- order(predicted_values_1, decreasing = TRUE)
ordered_actuals_1 <- test_1$isDefault[ordered_indices_1]

# Compute precision at different levels of K
k_values_1 <- seq(1, length(ordered_actuals_1), by = 100)  # Adjust step size as needed
precision_at_k_1 <- sapply(k_values_1, function(k) {
  predicted_positives_k_1 <- ordered_actuals_1[1:k]
  sum(predicted_positives_k_1 == 1) / k
})

# Plotting Precision at K
plot(k_values_1, precision_at_k_1, type = "o", pch = 19, col = "blue", xlab = "Top K cases", ylab = "Precision at K", main = "Precision at K Curve")
lines(k_values_1, precision_at_k_1, col = "blue")  # Connect points with lines

# 2
# Order the predictions by probability of being class 1
ordered_indices_2 <- order(predicted_values_2, decreasing = TRUE)
ordered_actuals_2 <- test_2$isDefault[ordered_indices_2]

# Compute precision at different levels of K
k_values_2 <- seq(1, length(ordered_actuals_2), by = 100)  # Adjust step size as needed
precision_at_k_2 <- sapply(k_values_2, function(k) {
  predicted_positives_k_2 <- ordered_actuals_2[1:k]
  sum(predicted_positives_k_2 == 1) / k
})

# Plotting Precision at K
plot(k_values_2, precision_at_k_2, type = "o", pch = 19, col = "blue", xlab = "Top K cases", ylab = "Precision at K", main = "Precision at K Curve")
lines(k_values_2, precision_at_k_2, col = "blue")  # Connect points with lines

# 3
# Order the predictions by probability of being class 1
ordered_indices_3 <- order(predicted_values_3, decreasing = TRUE)
ordered_actuals_3 <- test_3$isDefault[ordered_indices_3]

# Compute precision at different levels of K
k_values_3 <- seq(1, length(ordered_actuals_3), by = 100)  # Adjust step size as needed
precision_at_k_3 <- sapply(k_values_3, function(k) {
  predicted_positives_k_3 <- ordered_actuals_3[1:k]
  sum(predicted_positives_k_3 == 1) / k
})

# Plotting Precision at K
plot(k_values_3, precision_at_k_3, type = "o", pch = 19, col = "blue", xlab = "Top K cases", ylab = "Precision at K", main = "Precision at K Curve")
lines(k_values_3, precision_at_k_3, col = "blue")  # Connect points with lines


# 4
# Order the predictions by probability of being class 1
ordered_indices_4 <- order(predicted_values_4, decreasing = TRUE)
ordered_actuals_4 <- test_4$isDefault[ordered_indices_4]

# Compute precision at different levels of K
k_values_4 <- seq(1, length(ordered_actuals_4), by = 100)  # Adjust step size as needed
precision_at_k_4 <- sapply(k_values_4, function(k) {
  predicted_positives_k_4 <- ordered_actuals_4[1:k]
  sum(predicted_positives_k_4 == 1) / k
})

# Plotting Precision at K
plot(k_values_4, precision_at_k_4, type = "o", pch = 19, col = "blue", xlab = "Top K cases", ylab = "Precision at K", main = "Precision at K Curve")
lines(k_values_4, precision_at_k_4, col = "blue")  # Connect points with lines

# 5
# Order the predictions by probability of being class 1
ordered_indices_5 <- order(predicted_values_5, decreasing = TRUE)
ordered_actuals_5 <- test_5$isDefault[ordered_indices_5]

# Compute precision at different levels of K
k_values_5 <- seq(1, length(ordered_actuals_5), by = 100)  # Adjust step size as needed
precision_at_k_5 <- sapply(k_values_5, function(k) {
  predicted_positives_k_5 <- ordered_actuals_5[1:k]
  sum(predicted_positives_k_5 == 1) / k
})

# Plotting Precision at K
plot(k_values_5, precision_at_k_5, type = "o", pch = 19, col = "blue", xlab = "Top K cases", ylab = "Precision at K", main = "Precision at K Curve")
lines(k_values_5, precision_at_k_5, col = "blue")  # Connect points with lines


