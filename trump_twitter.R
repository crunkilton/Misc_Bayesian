## Cody Crunkilton
## Trump Tweets: Bayesian Dirichlet Multinomial Mixture Model / Correlation Network 
## 5/7/2018

setwd("C:/Users/Cody/Dropbox/1school17_18/bayes")

rm(list=ls())




# loading/cleaning data ---------------------------------------------------
library(tidyverse)
library(SnowballC)
library(tm)

## Load data from alex's website: 
con <- url('http://tahk.us/trump.rda')
load(con)
close(con)

# select tweets since he became president:
trump <- trump.president

## Remove punctuation
clean_text <- gsub("'", '', gsub("[^'#@A-Za-z0-9]", ' ', trump$text)) # get rid of apostrophes and symbols

## Create corpus
tweets <- VCorpus(VectorSource(clean_text))

## Add metadata
meta(tweets, "datetimestamp", type="local") <- trump$time
meta(tweets, "device", type="local") <- trump$device

## Making the document term matrix:
dtm_control <-
  list(tolower = T,
       removeNumbers = T,
       removePunctuation = F,
       stripWhitespace = T,
       stopwords = gsub("'", '', stopwords("SMART")),
       stemming = T)

trump.dtm <-
  DocumentTermMatrix(tweets, control = dtm_control)

# making the new DTM with words only used at least 50 times.

freq_dict <- findFreqTerms(trump.dtm, 50L, Inf)

trump.dtm <-
  DocumentTermMatrix(tweets,
                     control = append(dtm_control,
                                      list(dictionary=freq_dict)))

# the model ---------------------------------------------------------------

# to avoid having to run this: 
load("trump_topic_model.Rdata")


trump_jags <- "
model {

## Likelihood:

for (i in 1:N) {
y[i, 1:W] ~ dmulti(pi[topic[i], 1:W], n[i]) 
topic[i] ~ dcat(phi[1:T])
}

## Prior: 

for (t in 1:T) {
pi[t, 1:W] ~ ddirch(beta)
}
phi[1:T] ~ ddirch(alpha)

}
"
# running the model --------------------------------------------------------------

library(rjags)

# The data for the model:

tweet_data <- list(y = as.matrix(trump.dtm), # matrixing it
                   n = rowSums(as.matrix(trump.dtm)), # the number of individual words in the matrix
                   N = nrow(trump.dtm), # each document to loop over
                   W = ncol(trump.dtm), # each word to loop over
                   T = 10L) # number of topics

tweet_data$alpha <- rep(1, tweet_data$T) 
tweet_data$beta <- rep(1/10, tweet_data$W) 

# alpha and beta chosen because these are what others have used- no particular reason - If you had a reason to pick a different prior you could.

## Running and sampling the model: 
tweet_model <- jags.model(textConnection(trump_jags), data=tweet_data) # initializing the model

update(tweet_model, 1000) # running it 1000 times. ~10k would be better, but this is faster. 

jags.topics <- jags.samples(tweet_model, c("topic", "pi", "phi"), n.iter=1) # Sampling from posterior. Note only 1 iteration: we are essentially using this as an optimizer instead of averaging over the posterior like we would normally do.

save(jags.topics,
     file="trump_topic_model.RData",
     compress=TRUE)

## Most common words in each topic

most.common <- apply(jags.topics$pi, # all rows and columns from iteration 1 and chain 1
                     1, # apply to rows
                     function(x) freq_dict[order(x, decreasing=TRUE)[1:8]]) # take the top 8 words ordered by most common for each topic

jags.topics$phi # Topic probabilities, if you are interested...

# ggplot! ------------------------------------------------------------------

library(ggplot2) #if not loaded via tidyverse

topic.labels <- apply(most.common[1:3,], 2, paste, collapse="\n") # paste top three most common together to make labels for the topics

##### By device: 

# raw totals- less exciting because most is from iphone
device <- table(Device=factor(trump.president$device), Topic=jags.topics$topic) %>% 
  as.data.frame()

# as proportions: 
d_by_device <- device %>% 
  group_by(Device) %>% 
  mutate(Share=Freq/sum(Freq))

d_by_topic <- device %>% 
  group_by(Topic) %>% 
  mutate(Share=Freq/sum(Freq)) # didn't bother to plot this one - it's all Iphone. 

# The plot: 

ggplot()+
  geom_raster(data=d_by_device, aes(x=Topic, y=Device, fill=Share))+
  scale_x_discrete(labels=topic.labels, position="top")

# looks like the android tweets a lot about fake news and Russia and less about being honored about stuff

###### By time of day:

time <- table(Time=factor(format(trump.president$time, "%H")),
              Topic=jags.topics$topic) %>% 
  as.data.frame()


# as proportions by topic
df_by_time <- time %>% 
  group_by(Topic) %>% 
  mutate(Share_of_Topic=Freq/sum(Freq))


ggplot()+
  geom_raster(data=df_by_time, aes(x=Topic, y=Time, fill=Share_of_Topic))+
  scale_x_discrete(labels=topic.labels, position="top")

# Lots of tweeting starting at 6am...and 7am seems like the hotspot to hear about Russia

##### Future ideas: day of week, the entire time series, try out fewer/more topics. 



# Correlation Network ---------------------------------------------------------

# I've been going through a networks kick recently - thought I would see what a correlation network of the tweets would look like. 

library(igraph)

## just for fun: which words correlate with which?
findAssocs(trump.dtm, "big", .01)
findAssocs(trump.dtm, "fake", .01)

## Lets make this into a network problem: 

## creating correlation matrix
cors <- cor(as.matrix(trump.dtm))
cors <- ifelse(cors<0, 0, cors) # removing negative correlations
cors_1 <- ifelse(cors<.1, 0, cors) # removing correlations less than .1

## Making the graph objects: 
corgraph <- graph_from_adjacency_matrix(cors, mode="undirected", weighted=T)
corgraph1 <- graph_from_adjacency_matrix(cors_1, mode="undirected", weighted=T)

# Removing self correlations
corgraph <- simplify(corgraph) 
corgraph1 <- simplify(corgraph1) 

# Removing isolates for correlations >.1
corgraph1 <- induced.subgraph(corgraph1,
                              vids=which(igraph::degree(corgraph1)>1))#get rid of isolates

# Community detection: (only used walktrap here, could alsy try fast and greedy, louvain, infomap, etc)
walktrap <- cluster_walktrap(corgraph)
walktrap1 <- cluster_walktrap(corgraph1)

walktrap$membership # with all correlations: 4 groups (seems a bit small)
walktrap1$membership # looks like 9 groups for this one

## The plots: 
plot(corgraph, vertex.color=walktrap$membership, edge.width=E(corgraph)$weight*10)

plot(corgraph1, vertex.color=walktrap1$membership, edge.width=E(corgraph1)$weight*10)

## one last step, for a non-grapical representation:
library(tidyr)
library(tidygraph)
c <- as_tbl_graph(corgraph)
c1 <- as_tbl_graph(corgraph1)

topics <- c %>% 
  activate(nodes) %>% 
  mutate(topic=factor(walktrap$membership)) %>% 
  as.data.frame() %>% 
  arrange(topic)

topics

# Topic 1:
topics %>% 
  filter(topic==1) %>% 
  select(name)

# Topic 2:
topics %>% 
  filter(topic==2) %>% 
  select(name)

# Topic 3:
topics %>% 
  filter(topic==3) %>% 
  select(name)

# Topic 4:
topics %>% 
  filter(topic==4) %>% 
  select(name)

# This seems like too few topics. 

## Only using >.1 correlation:
topics1 <- c1 %>% 
  activate(nodes) %>% 
  mutate(topic=factor(walktrap1$membership)) %>% 
  as.data.frame() %>% 
  arrange(topic)

topics1 
# I can see how this makes sense. one topic is the stock market, another foreign policy, then immigration, then tax cuts/healthcare/legislation, etc. 


# Spinglass clustering for negative weights ---------------------------------------------------------------

## There aren't many clustering algorithms that deal well with negative correlations. One that does is spinglass, and below is what it thinks the communities are for the entire network: 

cor_all <- cor(as.matrix(trump.dtm))

corgraph_all <- graph_from_adjacency_matrix(cor_all, mode = "undirected", weighted=T) %>% 
  simplify()

spinglass <- cluster_spinglass(corgraph_all, implementation = "neg")

spinglass$membership # 6 groups?

spin <- as_tbl_graph(corgraph_all) %>% 
  activate(nodes) %>% 
  mutate(topic=factor(spinglass$membership)) %>% 
  as.data.frame() %>% 
  arrange(topic)

spin # looks pretty similar - seems interesting. 

# networks in this context is definitely less useful than the bayesian setup above, as this is just looking at what words co-occur the most, but still is interesting. It's more useful, though, when words can appear in multiple topics. 
