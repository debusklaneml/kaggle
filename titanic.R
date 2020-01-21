# Titanic - 1/20/2020 - M. L. DeBusk-Lane
library(tidyverse)
library(tidymodels)
library(tune)
library(janitor)
library(readxl)
library(tidylog)

# read in training dataset
train <- read_csv("train.csv") %>% clean_names() %>%
  select(-name, -ticket, -cabin) %>%
  mutate(female = if_else(sex == "female", 1, 0)) %>%
  mutate(survived = as.factor(survived)) %>%
  select(-sex)

# read in test dataset
test <- read_csv("test.csv") %>% clean_names() %>%
  select(-name, -ticket, -cabin) %>%
  mutate(female = if_else(sex == "female", 1, 0)) %>%
  select(-sex)

# data prep (recipe)
titanic_recipe <- train %>%
  recipe(survived ~ pclass + age + sib_sp + parch + fare + embarked) %>%
  step_dummy(embarked) %>% 
  step_corr(all_predictors()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  step_bagimpute(age, embarked_Q, embarked_S) %>%
  prep() 

# Test data
test_prepped <- titanic_recipe %>%
  bake(test)

# Training data
train_prepped <- juice(titanic_recipe)

# folds
folds <- vfold_cv(train, v = 10)

# model object:
xg <- boost_tree(
  mode = "classification",
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune()) %>%
  set_engine("xgboost") 

# check tunable parameters
xg %>% tunable()

grid <-  expand.grid(
  trees = c(50),
  min_n = c(3, 5, 7),
  tree_depth = 100,
  learn_rate = c(0.1, 0.5, 0.9))

# Workflow
titanic_wf <- workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(xg)
## Check it out
titanic_wf

# tune the grid
fits <- tune_grid(
  titanic_recipe,
  model = xg,
  resamples  = folds,
  grid = grid,
  metrics  = metric_set(roc_auc)
) 
# Mostly from (https://nyhackr.blob.core.windows.net/presentations/Totally_Tidy_Tuning_Tools-Max_Kuhn.pdf)
collect_metrics(fits)
show_best(fits)
good_values <- select_best(fits, maximize = FALSE)
new_recipe <- finalize_recipe(titanic_recipe, good_values)
new_mod <- finalize_model(xg, good_values)

# How do I tune this thing? 


# xgboost it!
# titanic_xg <- boost_tree(mode = "classification", trees = 110) %>%
#   set_engine("xgboost") %>%
#   fit(survived ~ ., data = train_prepped)

----------------------------------------------------------
# Tune it? 
titanic_xg.t <- tune_grid(survived ~ ., model = titanic_xg)

# Check predictions on the training dataset
titanic_xg %>%
  predict(train_prepped) %>%
  bind_cols(train_prepped) %>%
  metrics(truth = survived, estimate = .pred_class)

# Use the titanic_xg model on the test data and upload to Kaggle for scoring.... 
tit_xg_probs <- titanic_xg %>%
  predict(test_prepped, type = "prob") %>%
  bind_cols(test)

tit_xg_predict <- predict(titanic_xg, test_prepped, type = "prob") %>%
  bind_cols(predict(titanic_xg, test_prepped)) %>%
  bind_cols(test) %>%
  select(passenger_id, .pred_class) %>%
  rename(Survived = .pred_class) %>%
  rename(PassengerId = passenger_id) %>%
  glimpse()

write_csv(tit_xg_predict, "tit_xg_predict_v1.csv")
