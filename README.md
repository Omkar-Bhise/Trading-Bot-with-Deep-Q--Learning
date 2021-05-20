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

NLP : Sentiment140 training Data. The Input data contains sentiments in form of 0-Negative 2-Neutral and 4-Positive.We will convert these to 0,1 and 1.We will be using below 2 data Labels:
   
	Timestamp: Timestamp when the tweet was tweeted
   	Tweet : The actual tweet which we will analyze

4 Model :

4.1 TradingBot :

4.1.1 Model Description:
Model will take the current state of the environment (from the observation space) as input, and predict as output an action from the action space. We want the AI to initially explore the action space. Because Q value will be compared with probability epsilon, AI will choose a random action from the action space, not the best action from the action space. As Q value starts improving it will start to choose better action from action space. It's called the act method , and its epsilon greedy learning. The rest of the time it picks the network's best predicted action, simply by current state through the network to predict the best next action.
• Architecture : Deep Q Learning(ANN)
• Input: Open, High, Low, Close, Volume, Sentiment
• Output: Action(Buy, Sell, Hold)
• Loss Function :Mean Squared Error will calculate error between Q ap-
proximation made by neural network and actual Q calculated by Bellman
Equation.

4.1.2 Training Details :
The next step is to initialize the environment, including its observation and rewards. The model (Q function) predicts the highest probable action for the initial set of observations using argmax. The predicted action is passed through the environment and generates the new set of observations(s'), the reward for carrying the step and the done 
ag which indicates whether the episode is over (which involves reading through the entire training set). The memory is a buffer which keeps a record of the last 200 actions and its corresponding observations, rewards and the new set of observations obtained upon doing the action.The target is the Q table where we update the Q values for each set of action and its corresponding state according to the Bellman equation.

4.1.3 Hyper-parameter :
• Environment memory size [200]: Number of environments we are going to keep in memory
• Environment subset amount [20]: Amount of elements saved in memory which will be used for training
• Epsilon [1]: Epsilon of 1 in the beginning means we will do more exploration at rst and later on we do more exploitation
• Epsilon decrease [0.001] : We will decrease epsilon by this parameter
• Smallest possible epsilon [0.01] : This parameter decide how much exploitation happen
• Gamma [0.97] : Gamma determines how important the past is in experience replay
• Epoch [100]: Number of epochs we train the agent.
• Optimizer [adam]: The optimizer aims to update the Q values for minimizing the loss.

4.2 Natural Language Processing :

4.2.1 Implementation :
We used data from sentiment140 dataset. This data is fed to multinomial naiver bayes to convert two classes to three classes. The output of this model is then
fed to the BERT multiclass classifer which predicts three classes of tweets

4.2.2 Model :
We used below sequence of models to achieve our NLP result
• Multinomial Naive Bayes ALgorithm
• Multi-class classication using BERT model

4.2.3 Input :
Tweets about the stocks in the past 1 day from the timestamp window. Data
Labels (Timestamp of tweet and Tweet)

4.2.4 Output :
Sentiment about the stock (0: Negative 1: Neutral 2: Positive)

4.2.5 Training Details :
The training data in sentiment140 data set has tweets classifed in only two categories, so we initially used Naive Bayes algorithm on a fraction of our data
(1/100) to find probabilities of positive and negative tweets and then defined a predicted probability range of 0.4 - 0.6 for neutral tweets. Then we used this
model to classify our training data to have three categories in tweets using the predicted probability range. A BERT multiclass classifer was trained on this
data. After the model is created, we download twitter tweets for the date ranges for which the trading bot is to be trained. We use snscrape's python wrapper
to extract this data.We extract 100 tweets per day for all dates in the the given range. This data is then fed to the model generated by BERT and the mean
sentiment of each day is mapped.This data is then sent to train the trading bot.

5 Results Observations :

5.1 NLP:
We have achieved an accuracy of around 83 on training data of our Bert Model,the model input is the output of the Naive Bayes Model which has a prediction
accuracy of around 74 percent. 

5.2 Trading Bot Results:
We were able to achieve profits based on the model created by our proposed methodology.Below are the results for the three stocks we tested for.

6 Conclusion :
In this project, we intended to have a profitable trading strategy using financial indexes and twitter sentiment as an index.We successfully created a trading bot
using Deep Q-Networks based on ANN and achieved prots. Initially, we created a trading bot based on financial data and then integrated twitter sentiments and
achieved profitability.Although, there is scope for improvement in accuracy of our sentiment analysis model, yet our model is one of very few implementations
of a trading bot involving twitter sentiment as an index. In all the three stocks we backtested, we generated profits in order of thousand dollars

