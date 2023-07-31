set.seed(42)

#install.packages(" ")
library(lme4)
library(lmerTest)
library(readr)
library(tidyr)
library(data.table)
library(effectsize)
library(stringr)

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

path = file.path('R:/MIKADO_83820414/!Ergebnisse/mikado_fnirs_2_bids')
setting = "fNIRS_GLM_window_60.0"

##########################################################################################################
##########################################################################################################

# COEFFICIENTS: Contrasts
# Preparing your data
data <- read_csv2(file.path(path, "derivatives", "fnirs_preproc", setting, "nirs_glm_con.csv"), trim_ws = TRUE)

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
  model <- lmer(effect ~ -1 + ch_name + (1 | ID), data=subsetdata)
  
  anova(model)
  summary(model)
  table <- as.data.frame(coef(summary(model)))
  level = 1 - (0.05/ length(unique(subsetdata$ch_name)))
  confint <- as.data.frame(confint(model, method="boot", level = level, nsim=5000, boot.type="perc"))
  lower_ci = data.table("[0.025" = confint[[1]][3:nrow(confint)])
  upper_ci = data.table("0.975]" = confint[[2]][3:nrow(confint)])
  table <- cbind(table, lower_ci, upper_ci)
  write.csv(table, file.path(path, "derivatives", "fnirs_preproc", setting, paste("coefficients_", contrast, "_hbo.csv", sep="")))
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
  model <- lmer(effect ~ -1 + ch_name + (1 | ID), data=subsetdata)
  
  anova(model)
  summary(model)
  table <- as.data.frame(coef(summary(model)))
  level = 1 - (0.05/ length(unique(subsetdata$ch_name)))
  confint <- as.data.frame(confint(model, method="boot", level = level, nsim=5000, boot.type="perc"))
  lower_ci = data.table("[0.025" = confint[[1]][3:nrow(confint)])
  upper_ci = data.table("0.975]" = confint[[2]][3:nrow(confint)])
  table <- cbind(table, lower_ci, upper_ci)
  write.csv(table, file.path(path, "derivatives", "fnirs_preproc", setting, paste("coefficients_", contrast, "_hbr.csv", sep="")))
}
