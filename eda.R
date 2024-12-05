library(tidyverse)
library(lubridate)
library(scales)

options(scipen=999)

oneBoxPlot <- function(stock) {
  # be careful for errors
  fileName <- paste('Stock Data/', stock, '_historical_data.csv', sep='')
  data <- read.csv(fileName)
  
  ggplot(data=data, aes(x=Close)) + 
    geom_boxplot() + 
    theme_classic()
}

oneBoxPlot("VTABX")

data_comb <- (list.files(path = '~/Desktop/ITWS-4350-Project/Stock Data', pattern='*.csv', full.names=TRUE) 
     %>% lapply(function(file) {
       data <- read.csv(file)
       data$stockName <- sub(".*/([^/_]*)_.*", "\\1", file)
       data$Date <- as.Date(data$Date)
       return(data)
     }) 
     %>% bind_rows)

ggplot(data=data_comb, aes(x=Close)) + 
  geom_boxplot() + 
  facet_wrap( ~stockName, scales="free") + 
  theme_classic()


ggplot(data=data_comb, aes(x=Date)) + 
  geom_line(aes(y=Close), color="blue") + 
  facet_wrap(~stockName, scales="free") + 
  scale_y_continuous(name="Close", sec.axis=sec_axis(~./1, name="stock prices at close")) +
  geom_line(data=covid_dat, aes(x = date, y=new_cases_smoothed)) +

  theme_classic()

data_comb %>% group_by(stockName) %>% summarise(mean=mean(Close), median=median(Close), sd=sd(Close), range=max(Close)-min(Close))


covid_dat <- read.csv('COVID Data/filtered_us_covid_data.csv')

plot(covid_dat$date, covid_dat$new_cases_smoothed)
covid_dat$date <- as.Date(covid_dat$date)
ggplot(data=covid_dat, aes(x=date, y=new_cases_smoothed)) +
    geom_line(color="blue")+
    theme_classic() +
    labs(x= "Date", y = "Cases (smoothed)")

ggplot(data=covid_dat, aes(x=new_cases_smoothed)) +
  geom_boxplot()+
  theme_classic() +
  labs(x="Cases (smoothed)")

summary(covid_dat$new_cases_smoothed)
