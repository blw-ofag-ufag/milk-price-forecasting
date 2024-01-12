#===============================================================================================================
# title: Milch price forecasting model
# author: Damian Oswald
# date: 2023-11-08
#===============================================================================================================

#===============================================================================================================
# Setting up the work space
#===============================================================================================================

# user inputs
name_of_the_data <- "milk-price-data.xlsx"

# define target variable
target <- "CH_Milchpreis"

# how many months of the data set should be used for the test set?
test_set_size <- 18

# define feature names (all of these features are used to predict `target` using machine learning)
features <- c("TotMilch_Menge_inkl_LI", "CH_Milchpreis", "Molkerei_Milchpreis",
              "Verkaest_Milchpreis", "Gewerblich_Milchpreis", "Bio_Milchpreis",
              "EU_Milchpreis_CHF", "Kurs_CHF_Euro", "Total_Staatlich_Zahlung",
              "staatlich_kg_Milch", "Kontingent_01", "Freihandel_01", "EU_Politik_01",
              "Industriell_verkäst_Milchpreis", "Import_Menge_Käse_und_Quark_kg",
              "Schlachtungen_Kühe", "Magermilchpulver_Prod_Menge",
              "Vollmilchpulver_Prod_Menge", "Butter_Prod_Menge",
              "Export_Menge_Käse_und_Quark_kg", "Milchpreis_NZ_Euro")

# conditional installation of packages (if allready installed, just add package to search path)
for (package in c("forecast", "ggplot2", "tidyr", "GGally", "tibble", "magrittr", "readxl", "visdat", "seastests", "ggcorrplot", "openxlsx", "glmnet", "lubridate")) {
  if(!do.call(require,list(package))) install.packages(package)
  do.call(library,list(package))
}

# print out an introductory message in the console
cat("\f\nFederal Office of Agriculture (2023)\n___  ____ _ _     ______     _           ______                           _   _             \n|  \\/  (_) | |    | ___ \\   (_)          |  ___|                         | | (_)            \n| .  . |_| | | _  | |_/ / __ _  ___ ___  | |_ ___  _ __ ___  ___ __ _ ___| |_ _ _ __   __ _ \n| |\\/| | | | |/ / |  __/ '__| |/ __/ _ \\ |  _/ _ \\| '__/ _ \\/ __/ _` / __| __| | '_ \\ / _` |\n| |  | | | |   <  | |  | |  | | (_|  __/ | || (_) | | |  __/ (_| (_| \\__ \\ |_| | | | | (_| |\n\\_|  |_/_|_|_|\\_\\ |_|  |_|  |_|\\___\\___  |_| \\___/|_|  \\___|\\___\\__,_|___/\\__|_|_| |_|\\__, |\n                                                                                       __/ |\n                                                                                      |___/\n")

# read the data from the excel file directly
data <- name_of_the_data |>
  readxl::read_excel() |>
  as.data.frame() |>
  subset(select = features)

# export (or override) data as csv
write.csv(data, file = "milk-price-data.csv", row.names = FALSE)

# convert data frame to time series object
data_ts <- ts(data, start = 2000, frequency = 12)

# change the variable `TotMilch_Menge_inkl_LI` from monthly total to daily production
# (this yields an annual cycle more representative for the actual milk production independent of monthly length)
paste(floor(time(data_ts)),cycle(data_ts),"1",sep="-") |>
  as.Date() |>
  lubridate::days_in_month() -> days_in_month
data[,1] <- data[,1]/days_in_month
data_ts[,1] <- data_ts[,1]/days_in_month


#===============================================================================================================
# Exploratory data analysis
#===============================================================================================================

# function to convert a decomposed time series object to a data frame
ts2df <- function(x) {
  d <- decompose(x)
  data.frame(year = floor(time(x)), month = as.integer(cycle(x)),
             observation = d$x,
             trend = d$trend,
             seasonal = seasonal(d),
             remainder = d$random)
}

# conditional creation of the output folder
if(!file.exists(file.path(getwd(),"output"))) dir.create(file.path(getwd(),"output"))

# create empty pdf
pdf(file.path(getwd(),"output","time-series-decomposition.pdf"), width = 16, height = 9)

# title slide
plot.new()
text(0, 1, paste("This PDF contains results from an automated time-series analysis report."), pos = 4, xpd = NA)
text(0, 0.95, "Find more detailed information on https://github.com/blw-ofag-ufag/milk-price-forecasting", pos = 4, xpd = NA)
text(0, 0.9, Sys.time(), pos = 4, xpd = NA)
text(0, 0.85, version$version.string, pos = 4, xpd = NA)

# missing values
data_ts |> as.data.frame() |> visdat::vis_miss() |> print()

# correlation plot
data_ts |> cor(use = "complete.obs") |> ggcorrplot::ggcorrplot(method = "circle", type = "lower", outline.color = par()$fg, lab_size = 2, lab = TRUE) |> print()
openxlsx::write.xlsx(x = as.data.frame(cor(data_ts, use = "complete.obs")), file = file.path("output","correlations.xlsx"))

# seasonal decomposition of the time series data
for (i in 1:length(features)) {
  cat("\rDecomposing time series [", i, "/", length(features),"]", sep = "")
  if(!seastests:::isSeasonal(data_ts[,features[i]])) {
    data_ts[,features[i]] |> forecast:::autoplot.ts(ylab = "") + ggtitle(paste(features[i], ": Time series plot"), subtitle = paste0("Ollech and Webel's combined seasonality test didn't detect seasonality in this data (p-value = ",paste(round(max(seastests::combined_test(data_ts[,features[i]])$Pval),3),collapse = ", "),")")) -> p
    print(p)
  } else {
    data_ts[,features[i]] |> decompose() -> decomposition
    decomposition |> forecast:::autoplot.decomposed.ts() + ggtitle(paste(features[i],": Decomposition of additive time series"), "Time series decomposition into seasonal, trend and irregular components using moving averages.") -> p
    print(p + theme_minimal())
  }
}

# close pdf
dev.off()

# write all the results into one excel sheet
list <- lapply(features, function(i) data_ts[,i] |> ts2df() |> signif(digits = 5))
names(list) <- features
write.xlsx(list, file = file.path("output","decomposition.xlsx"))

#===============================================================================================================
# Prepare data for forecasting
#===============================================================================================================

cat("\nPreparing data")

# Add dummy variables for each month (to model monthly effects)
Months <- model.matrix(~ as.factor(cycle(data_ts)) - 1)
colnames(Months) <- month.name

# sine and cosine transformer
sinCosTransform <- function(x, frequency = 1) {
  cbind(sin = sin(x * 2 * pi/frequency), cos = cos(x * 2 * pi/frequency))
}

# function to add lag to a numeric vector x (can generate multiple lags at the same time)
lag <- function(x, k = 1, col.names = NA) {
  Y <- sapply(k, function(k) {
    n <- length(x)
    c(x[(1+k):n], rep(NA, k)) 
  })
  if(!is.na(col.names)) colnames(Y) <- col.names
  else colnames(Y) <- paste0("y", k)
  return(Y)
}

# add transformations and labels (with a lag)
df <- cbind(lag(data_ts[,target], k = 1:3),
            time = as.numeric(time(data_ts)),
            sinCosTransform(as.numeric(time(data_ts))),
            Months,
            as.matrix(as.data.frame(data_ts)))

# replace all NA value by zero
df[,features][is.na(df[,features])] <- 0

# scale the data (IMPORTANT: apply the same normalization to y1, y2, and y3; otherwise re-scale is wrong!)
df <- cbind((df[,1:3] - mean(data[,target]))/sd(data[,target]), time = df[,4], scale(df[,-c(1:4)]))

# function to re-scale scaled values to the original size
rescale <- function(z, mu = mean(data[,target]), sigma = sd(data[,target])) sigma*z + mu

# test-train-split (keep the last year as a test dataset)
df_train <- head(df, -test_set_size)
df_test <- tail(df, test_set_size)

# In the train dataset, divide folds for cross validation as six-month periods 
foldid <- (df_train[,"time"] %/% 0.5 - 3999)

# Save all features allowed to predict the model in one vector
features <- c("sin","cos",month.name,features)
label <- "y1"


#===============================================================================================================
# Forecasting the milk price with lasso and ridge regression
#===============================================================================================================

# we want to forecast 1, 2 and 3 months into the future
for (i in 3:1) {
  
  # print message
  cat("\rFitting regularized least squares [", (4-i), "/3]", sep = "")
  
  # traine one L1 and one L2 regularized least squares model (lasso and ridge)
  L1 <- glmnet::cv.glmnet(x = df_train[,features], y = df_train[,paste0("y",i)], lambda = exp(seq(-12,0,0.25)), alpha = 1, foldid = foldid)
  L2 <- glmnet::cv.glmnet(x = df_train[,features], y = df_train[,paste0("y",i)], lambda = exp(seq(-12,0,0.25)), alpha = 0, foldid = foldid)
  
  # visualize the coefficients as a coefficients plot
  coefficients <- data.frame(Variable = rep(rownames(coefficients(L1)), 2), Coefficient = c(as.numeric(coef(L1)),as.numeric(coef(L2))), Norm = rep(c("L1", "L2"), each = length(coef(L2))))
  coefficients$Variable <- factor(coefficients$Variable, levels = rownames(coefficients(L2))[order(abs(rowSums(cbind(coef(L1), coef(L2)))))])
  highlight <- coefficients[coefficients$Variable %in% rownames(coef(L1))[which(abs(coef(L1))>0)],]
  assign(paste0("plot_coefficients_",paste0("y",i)),
         ggplot2::ggplot(coefficients, aes(Coefficient, Variable)) +
           geom_line(aes(group = Variable), alpha = 0.3) + geom_point(aes(color = Norm), size = 2, alpha = 0.5) +
           geom_line(data = highlight, aes(group = Variable)) + geom_point(data = highlight, aes(color = Norm), size = 3)
  )
  
  # save predictions
  assign(paste0("predictions_",paste0("y",i)), {
    data.frame(
      Lasso = as.numeric(predict(L1, newx = as.matrix(df_test[,features]))),
      Ridge = as.numeric(predict(L2, newx = as.matrix(df_test[,features])))
    )
  })
}

#===============================================================================================================
# Forecasting the milk price with (seasonal) ARIMA
#===============================================================================================================

# find ARIMA(1,1,1) model to fit the data
predictions_arima <- matrix(NA, test_set_size, 3)
cat("\n")
for (t in 1:test_set_size) {
  cat("\rFitting ARIMA [", t, "/", test_set_size, "]", sep = "")
  df[1:(nrow(df)-t),target] |>
    arima(order = c(1,1,1)) |>
    predict(n.ahead = 3, se.fit = FALSE) -> predictions_arima[test_set_size+1-t,]
}

# find ARIMA(1,1,1)(1,1,1)[12] model to fit the 
predictions_sarima <- matrix(NA, test_set_size, 3)
cat("\n")
for (t in 1:test_set_size) {
  cat("\rFitting SARIMA [", t, "/", test_set_size, "]", sep = "")
  df[1:(nrow(df)-t),target] |>
    arima(order = c(1,1,1), seasonal = list(order = c(1,1,1), period = 12)) |>
    predict(n.ahead = 3, se.fit = FALSE) -> predictions_sarima[test_set_size+1-t,]
}

# Combine with RLS predictions
predictions_y1 <- cbind(predictions_y1, ARIMA = predictions_arima[,1], SARIMA = predictions_sarima[,1])
predictions_y2 <- cbind(predictions_y2, ARIMA = predictions_arima[,2], SARIMA = predictions_sarima[,2])
predictions_y3 <- cbind(predictions_y3, ARIMA = predictions_arima[,3], SARIMA = predictions_sarima[,3])

# Combine all forecastings (with unseen observations to compare)
t <- max(df[,"time"]) + c(1/12, 2/12, 3/12)
forecastings <- rescale(rbind(t1 = tail(predictions_y1, 1), t2 = tail(predictions_y2, 1), t3 = tail(predictions_y3, 1)))
write.xlsx(data.frame(Year = t%/%1, Month = t%%1*12+1, forecastings), file = file.path("output","predictions.xlsx"))


#===============================================================================================================
# Testing all models
#===============================================================================================================

# calculate root mean squared error (RMSE)
rmse <- function(predictions, observations) {
  Errors <- (rescale(predictions) - rescale(observations))
  sqrt(colMeans(Errors^2, na.rm = TRUE))
}
RMSE <- sapply(1:3, function(i) rmse(get(paste0("predictions_y",i)), df_test[,paste0("y",i)])) |> t()
RMSE <- data.frame(horizon = rep(as.character(1:3), times = 4), gather(as.data.frame(RMSE)))
RMSE[,"value"] <- RMSE[,"value"] * 0.01 # change from cents to CHF

# make plot of all model performances
overall_comparison <- ggplot(data = RMSE, aes(x = key, y = value, fill = horizon)) +
  ylab("Root mean squared error [CHF]") +
  geom_bar(stat = "identity", position=position_dodge()) +
  geom_text(aes(label = round(value, 3)), color = "black", position = position_dodge(0.9), vjust = -1) +
  scale_fill_brewer(palette = "Reds") +
  ggtitle("Model performance comparison",paste("This chart shows the root mean squared error (RMSE), which represents the average error, of different models.\nThe models were all tested on the milk price data of the past",test_set_size,"months, but trained without that data."))

#===============================================================================================================
# Export all plots to one PDF
#===============================================================================================================

# print out message
cat("\nGenerate report [1/1]")

# Export all plots as one PDF
pdf(file.path("output","machine-learning-report.pdf"), width = 16, height = 9)

# title slide
plot.new()
text(0, 1, paste("This PDF contains results from an automated machine learning model to forecast `", target, "`.", sep = ""), pos = 4, xpd = NA)
text(0, 0.95, "Find more detailed information on https://github.com/blw-ofag-ufag/milk-price-forecasting", pos = 4, xpd = NA)
text(0, 0.9, Sys.time(), pos = 4, xpd = NA)
text(0, 0.85, version$version.string, pos = 4, xpd = NA)

# Forecast of unseen values
plot(x = df_test[,"time"], y = rescale(df_test[,target]), xlim = range(df_test[,"time"]) + c(0, 1/2), axes = FALSE, pch = 16, type = "o", las = 1, lwd = 3, ylab = "Swiss milk price [CHF/100]", xlab = "Time")
grid(lty = 1)
points(x = df_test[,"time"], y = rescale(df_test[,target]), pch = 16)
lines(x = df_test[,"time"], y = rescale(df_test[,target]), lwd = 2)
last <- max(df_test[,"time"])
colors = c(ARIMA = "#00BFC4", SARIMA = "#00BFC4", Lasso = "#F8766D", Ridge = "#F8766D")
for (model in colnames(forecastings)) {
  lines(x = max(df_test[,"time"]) + c(1/12, 2/12, 3/12), y = forecastings[,model], lwd = 3, col = colors[model])
  text(x = max(df_test[,"time"]) + 3/12, y = forecastings[3,model], labels = model, col = colors[model], pos = 4)
}
axis(1, lwd = NA, cex.axis = 0.8, at = 2000:2050, labels = 2000:2050)
axis(2, lwd = NA, cex.axis = 0.8, las = 1)

# Model comparison
print(overall_comparison + theme_minimal())

# Pairs panel
for(i in 1:3) {
  cbind(Observation = df_test[,paste0("y",i)], get(paste0("predictions_",paste0("y",i)))) |>
    na.omit() |>
    rescale() |>
    ggpairs(lower = list(continuous = "smooth"), progress = FALSE) +
    theme_minimal() +
    ggtitle(paste0("Comparing the forecast values (t + ",i,")"), subtitle = paste("These plots compare the forecast milk prices with the observed ones of the past", test_set_size, "months.")) -> plot
  print(plot)
}

# Coefficients of lasso and ridge regression
for (i in 1:3) print(get(ls()[grep("plot_coefficients",ls())][i]) + ggtitle(paste0("Visualization of the regularized least squares coefficients (t + ", i,")"), subtitle = "This coefficients are retrieved from the best-fit models trained with the L1-norm (lasso regression) and L2-norm (ridge regression). They indicate relative variable importance.") + theme_minimal())

# Details of lasso and ridge
par(mfrow = c(2,2), mar = c(4,4,5,1)+0.1)
plot(L1, col = "black"); mtext("Lasso regression (L1)", line = 3, font = 2); abline(v = L1$lambda.min)
plot(L2); mtext("Ridge regression (L2)", line = 3, font = 2)
(L1$glmnet.fit) |> plot(col = "red", xvar = "lambda"); mtext("Lasso regression (L1)", line = 3, font = 2); abline(v = L1$lambda.min)
(L2$glmnet.fit) |> plot(col = "red", xvar = "lambda"); mtext("Ridge regression (L2)", line = 3, font = 2)

dev.off()

cat("\n\nDone!")
rm(list = ls())
