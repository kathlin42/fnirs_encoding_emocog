set.seed(42)

#install.packages(" ")
library(lme4)
library(lmerTest)
library(readr)
library(tidyr)
library(data.table)
library(effectsize)
library(stringr)
library(dplyr)
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

path = file.path('R:/MIKADO_83820414/!Ergebnisse/mikado_fnirs_2_bids')
include_silence = "_correct_silence" #_correct_silence _include_silence
include_hand = "" #with_handedness_
setting = paste("fNIRS_GLM_window_60.0", include_silence, sep="")
save = file.path(path, "derivatives", "fnirs_preproc", paste(include_hand, setting, sep = ""), "LMM_coefficients")
if (file.exists(save)) {
  cat("The save folder already exists")
} else {
  dir.create(save)
}
nsim = 5000
alpha = 0.05
##########################################################################################################
##########################################################################################################

# COEFFICIENTS: Contrasts
# Preparing your data
data <- read_csv2(file.path(path, "derivatives", "fnirs_preproc", setting, "nirs_glm_con.csv"), trim_ws = TRUE)
if (include_hand == "with_handedness_"){
  demographics <- read_csv2(file.path(path, "sourcedata", "demographics", "demographics_scores.csv"), trim_ws = TRUE)
  # Rename old_column1 to new_column1
  colnames(demographics)[colnames(demographics) == "Subject ID"] <- "ID"
  # Select column1 and column2
  demographics <- demographics[, c("ID", "hand")]
  data <- left_join(data, demographics, by = "ID")}

## HBO
contrasts <- c(unique(data$Contrast))
for (contrast in contrasts){
  subsetdata <- subset(data, Chroma == 'hbo')
  subsetdata <- subset(subsetdata, Contrast == contrast)
  hist(subsetdata$effect)
  
  # Drop NAs
  subsetdata <- subsetdata %>% drop_na(effect)
  
  # Standardization
  subsetdata$effect = (subsetdata$effect - mean(subsetdata$effect)) / sd(subsetdata$effect)
  hist(subsetdata$effect)
  
  # Model
  if (include_hand == "with_handedness_") {
    model <- lmer(effect ~ -1 + ch_name + (1 | ID) + hand, data=subsetdata)
  } else {
    model <- lmer(effect ~ -1 + ch_name + (1 | ID), data=subsetdata)
  }
  
  
  anova(model)
  summary(model)
  table <- as.data.frame(coef(summary(model)))
  level = 1 - (alpha/ length(unique(subsetdata$ch_name)))
  confint <- as.data.frame(confint(model, method="boot", level = level, nsim=nsim, boot.type="perc"))
  lower_ci = data.table("[0.025" = confint[[1]][3:nrow(confint)])
  upper_ci = data.table("0.975]" = confint[[2]][3:nrow(confint)])
  table <- cbind(table, lower_ci, upper_ci)
  write.csv(table, file.path(save, paste("coefficients_", include_hand, contrast, "_hbo.csv", sep="")))
}

## HBR
for (contrast in contrasts){
  subsetdata <- subset(data, Chroma == 'hbr')
  subsetdata <- subset(subsetdata, Contrast == contrast)
  hist(subsetdata$effect)
  
  # Drop NAs
  subsetdata <- subsetdata %>% drop_na(effect)
  
  # Standardization
  subsetdata$effect = (subsetdata$effect - mean(subsetdata$effect)) / sd(subsetdata$effect)
  hist(subsetdata$effect)
  
  # Model
  if (include_hand == "with_handedness_") {
    model <- lmer(effect ~ -1 + ch_name + (1 | ID) + hand, data=subsetdata)
  } else {
    model <- lmer(effect ~ -1 + ch_name + (1 | ID), data=subsetdata)
  }
  
  anova(model)
  summary(model)
  table <- as.data.frame(coef(summary(model)))
  level = 1 - (alpha/ length(unique(subsetdata$ch_name)))
  confint <- as.data.frame(confint(model, method="boot", level = level, nsim=nsim, boot.type="perc"))
  lower_ci = data.table("[0.025" = confint[[1]][3:nrow(confint)])
  upper_ci = data.table("0.975]" = confint[[2]][3:nrow(confint)])
  table <- cbind(table, lower_ci, upper_ci)
  write.csv(table, file.path(save, paste("coefficients_", include_hand, contrast, "_hbr.csv", sep="")))
}
#####################################################################################################
ch_data <- read_csv2(file.path(path, "derivatives", "fnirs_preproc", setting, "nirs_glm_cha.csv"), trim_ws = TRUE)
ch_data <- subset(ch_data, ch_name == 'S9_D11 hbr')
ch_data$Emotion_Condition = ""
ch_data$Load_Condition = ""
setDT(ch_data)[Condition == "HighSil", `:=`(Emotion_Condition = "Silence", Load_Condition = "High")]
setDT(ch_data)[Condition == "HighNeu", `:=`(Emotion_Condition = "Neutral", Load_Condition = "High")]
setDT(ch_data)[Condition == "HighNeg", `:=`(Emotion_Condition = "Negative", Load_Condition = "High")]
setDT(ch_data)[Condition == "HighPos", `:=`(Emotion_Condition = "Positive", Load_Condition = "High")]
setDT(ch_data)[Condition == "LowSil", `:=`(Emotion_Condition = "Silence", Load_Condition = "Low")]
setDT(ch_data)[Condition == "LowNeu", `:=`(Emotion_Condition = "Neutral", Load_Condition = "Low")]
setDT(ch_data)[Condition == "LowNeg", `:=`(Emotion_Condition = "Negative", Load_Condition = "Low")]
setDT(ch_data)[Condition == "LowPos", `:=`(Emotion_Condition = "Positive", Load_Condition = "Low")]
ch_data <- subset(ch_data, Emotion_Condition == 'Positive' | Emotion_Condition == 'Negative')
# Drop NAs
ch_data <- ch_data %>% drop_na(theta)
# Standardization
ch_data$theta = (ch_data$theta - mean(ch_data$theta)) / sd(ch_data$theta)
hist(ch_data$theta)
model <- lmer(theta ~ Load_Condition * Emotion_Condition + (1 | ID),
                             data=ch_data)
anova(model)
eta_squared(model)
summary(model)
table <- as.data.frame(coef(summary(model)))
level = 1 - alpha
confint <- as.data.frame(confint(model, method="boot", level = level, nsim=nsim, boot.type="perc"))
lower_ci = data.table("[0.025" = confint[[1]][3:nrow(confint)])
upper_ci = data.table("0.975]" = confint[[2]][3:nrow(confint)])
table <- cbind(table, lower_ci, upper_ci)
write.csv(table, file.path(path, "derivatives", "fnirs_preproc", paste("REANALYZE_S9_D11_hbr_coefficients_", include_hand, 'contrast_inter_EMO_NegPos', ".csv", sep="")))

