data("mtcars")
View(mtcars)

# class column will be am

mtcars$am <- factor(mtcars$am , levels = c(0,1) , labels = c("Auto" , "Man"))
mtcars$am

normalize <- function(x){
  return ((x - min(x)) / (max(x)) - min(x))
}

# normalized 
mtcars_n = as.data.frame(lapply(mtcars[,-9],normalize))
mtcars_n


mtcars_train <- mtcars_n[1:20 , ]
mtcars_test <- mtcars_n[21:32, ]

mtcars_train_label <- mtcars_n