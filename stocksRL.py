#Importing Libraries
import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt     
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features
from gym_anytrading.envs import StocksEnv


#reading SPX csv file and pre-processing
df = pd.read_csv("SPX.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date",ascending=True,inplace=True)
df.set_index("Date", inplace=True)

df["Open"] = df["Open"].apply(lambda x: float(x.replace(',','')))
df["High"] = df["High"].apply(lambda x: float(x.replace(',','')))
df["Close"] = df["Close"].apply(lambda x: float(x.replace(',','')))
df["Low"] = df["Low"].apply(lambda x: float(x.replace(',','')))

#Creating Environment
env = gym.make("stocks-v0",df=df,frame_bound=(5,200),window_size=5)
#env.signal_features
#env.action_space

#Taking Random actions in env
state = env.reset()
while True:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

#Creating model and learning from environment 
env_training = lambda: gym.make('stocks-v0', df=df, frame_bound=((5,200)),window_size=5)
env = DummyVecEnv([env_training])

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

#Visualizing
env = gym.make('stocks-v0',df=df, frame_bound = ((200,253)),window_size=5)
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, status = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

#Add TA to data
data = pdr.get_data_yahoo('SPY', start='2017-01-01',end='2021-10-01')

df2 = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume',fillna=True)
pd.set_option('display.max_columns', None)

#Creating environment with technical indicators
def my_processed_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:,'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ["Close", "Volume", "momentum_rsi", "volume_obv", "trend_macd_diff"]].to_numpy()[start:end]
    return prices,signal_features

class MyCustomEnv(StocksEnv):
    _process_data = my_processed_data

env2 = MyCustomEnv(df=df2, window_size=5, frame_bound=(5,700))
# env2.signal_features

#Visualizing Agent in TA environment
training_env = lambda: env2
env = DummyVecEnv([training_env])

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

env = MyCustomEnv(df=df2,  window_size=5, frame_bound=(700,1000))
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, status = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()