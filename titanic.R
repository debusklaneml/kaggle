# Titanic - 1/20/2020 - M. L. DeBusk-Lane
library(tidyverse)
library(tidymodels)
library(janitor)
library(readxl)
library(tidylog)




# read in training dataset
train <- read_csv("train.csv") %>% clean_names() %>%
  select(-passenger_id, -name, -ticket, -cabin) %>%
  mutate(female = if_else(sex == "female", 1, 0)) %>%
  mutate(survived = as.factor(survived)) %>%
  select(-sex)

# read in test dataset
test <- read_csv("test.csv") %>% clean_names() %>%
  select(-passenger_id, -name, -ticket, -cabin) %>%
  mutate(female = if_else(sex == "female", 1, 0)) %>%
  select(-sex)

# data prep 
titanic_recipe <- train %>%
  recipe(survived ~ .) %>%
  step_dummy(embarked) %>% 
  step_corr(all_predictors()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  step_bagimpute(age, embarked_Q, embarked_S) %>%
  prep()

test_prepped <- titanic_recipe %>%
  bake(test)

train_prepped <- juice(titanic_recipe)

# xgboost it!
titanic_xg <- boost_tree(mode = "classification", trees = 100) %>%
  set_engine("xgboost") %>%
  fit(survived ~ ., data = train_prepped)

# Check predictions on the training dataset
titanic_xg %>%
  predict(train_prepped) %>%
  bind_cols(train_prepped) %>%
  metrics(truth = survived, estimate = .pred_class)


