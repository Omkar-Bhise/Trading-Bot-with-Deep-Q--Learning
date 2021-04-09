# Trading-Bot-with-Deep-Q--Learning

Team Member:
Omkar Bhise, Shashank Shekhar

Problem Statement: 

Financial markets are very dynamic in nature. Trading in the current time is far different than 10 to 15 years ago. Today's markets are very turbulent in nature with increased volatility, periodical flash crashes and unpredictable world events like covid -19. Such factors have influenced the way trade. Result of such influence is a short lived pattern in historical data which is hard to find.Most of the current solutions are based on Machine Learning.The dynamic nature of data and requirement of large labelled datasets adversely affects the ML models. DRL doesn’t rely on large labelled data sets and, therefore, has a better feasibility than ML models.DRL works on a reward function and hence optimises according to the dynamic nature of financial markets.

Literature Survey:

	Data Sources: 

	https://github.com/alpacahq/alpaca-trade-api-python
	https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit
	https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets

	Existing work: 

	https://towardsdatascience.com/24x5-stock-trading-agent-to-predict-stock-prices-with-deep-learning-with-deployment-c15570720ae9
	http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26644050.pdf

Dataset:

Deep Q-Learning :This dataset will be used for Trading bot training. Which will be extracted from proprietary data sources like Alpaca Stock Brokerage. Data extracted from Alpaca Stock Brokerage will contain column as following 
	
	Timestamp : Date and time one which following data is being collected for that stock.
   Open : Opening value of stock for that particular timestamp.
   High:  Highest stock value for that particular timestamp.
   Low: Lowest stock value for that particular timestamp.
   Close: Closing value of stock for that particular timestamp
   Volume: Volume of trades happening of stock for that particular timestamp.

NLP : Sentiment140 training Data. The Input data contains sentiments in form of 0-Negative 2-Neutral and 4-Positive.We will convert these to 0,0.5 and 1.We will be using below 2 data Labels:
   
	Timestamp: Timestamp when the tweet was tweeted
   Tweet : The actual tweet which we will analyze

Model :
   TradingBot:

		Implementation: Model will take the current state of the environment (from the observation space) as input, and predict as output an action from the action space. We want the AI to initially explore the action space. Because Q value will be compared with probability epsilon, AI will choose a random action from the action space, not the best action from the action space. As Q value starts improving it will start to choose better action from action space. It's called the act method , and its epsilon greedy learning. The rest of the time it picks the network’s best predicted action, simply by current state through the network to predict the best next action.
		Model Description : Double Deep Q Learning (ANN / LSTM), Deep Q Learning(ANN / LSTM)
		Input: Open, High, Low, Close, Volume, Sentiment
		Output: Action(Buy, Sell, Hold)

	NLP:
	
		Implementation : We will firstly use word2vec embedding to convert text data to numerical data. This data will be fed to Deep Neural Network models. We will use crossEntropy loss in both the models. We will then choose the best performing model based on accuracy and F1 score.
		Model : Text CNN Model for multiclass classification , Multiclass classification using BERT model
		Input : Tweets about the stocks in the past 1 hour from the timestamp window. Data Labels (Timestamp of tweet  and Tweet)
		Output : Sentiment about the stock (0: Negative 0.5: Neutral 1: Positive)


Project Outcome:

We intend to create a trading bot based on Deep Q-Learning with intended profit as output. Along with input signals containing stock data, we also intend to create an additional input signal to this bot which uses an NLP based sentiment analysis model.This model uses new data(twitter, etc.) to calculate sentiments for a particular stock for that timestamp and hence assist our trading bot in better outputs.

Challenges:

Current model is considering the entire portfolio for that particular stock which is not an ideal way for trading bot to work. For best action, bot should be considering the percentage of portfolio that should be considered for action space. For example, if someone has 10000$ in stock of apple it should be able to sell the most optimized amount of that stock. 

Reward function engineering is another problem, since we are only rewarding trading bot for profit but not punishing him for losses. Therefore, some sort of reward function engineering is required.

Even Though, we have done some testing we are still not understanding which neural network we should use for Q value approximation. 

Feature engineering we are only considering signals provided by proprietary API of Alpaca.However, more signals generation can be done with given signals. Additionally, We can consider more external signals like sentiment signals.    
