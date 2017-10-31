sept = read.csv('WeatherData.csv',stringsAsFactors = FALSE)
sept$diff = sept$high - sept$low

sept$Date = strptime(sept$Oct,"%m/%d/%y")


library(ggplot2)


add_degree <- function(x) {
  paste(x, " degrees",sep = "")
} 

add_date <- function(x)
{
  print(x)
  return(paste0(x,'17'))
}


ggplot(data = sept, aes(x = Date, y = diff)) + geom_line(group=1,color = 'red') + xlab("Dates") +ylab("Temperatures")+
  theme(axis.text.x = element_text(angle=270), axis.text.y = element_text(angle=45) ) + scale_y_continuous(labels = add_degree, limits= c(9,21))+scale_x_date(labels = add_date)

