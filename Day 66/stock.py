import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas as pd
from pandas_datareader import data as wb

stocks = [
    {
        'ticker': 'RELIANCE.NS',
        'name': 'Reliance Industries'
    },
    {
        'ticker': 'TCS.NS',
        'name': 'Tata Consultancy Services'
    },
    {
        'ticker': 'VMW',
        'name': 'VMWare'
    }
]

def create_plots(stocks):
    data = pd.DataFrame()
    for stock in stocks:
        data[stock['ticker']] = wb.DataReader(stock['ticker'], data_source='yahoo', start='2019-11-1')['Adj Close']
    returns = data.apply(lambda x: (x / x[0] * 100))

    plt.figure(figsize=(10,10))
    for stock in stocks:
        plt.plot(returns[stock['ticker']], label=stock['name'])
    plt.legend()
    plt.ylabel('Cumulative Returns %')
    plt.xlabel('Time')
    plt.show()
create_plots(stocks)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def updatedPrice(oldPrice, mean, std):
    """
    oldPrice: existing price of stock
    mean: average of data
    std: standard deviation of data
    """
    r = np.random.normal(0,1,1)
    currentPrice = oldPrice + oldPrice*(mean/255+std/np.sqrt(324)*r)
    return currentPrice

z = np.random.normal(0,1,255)
u = 0.1
sd = 0.3
counter = 0
price = [100]
t = [0]

def animate(i):
    global t,u,sd,counter
    x = t
    y = price
    counter+=1
    x.append(counter)
    y.append(updatedPrice(price[counter-1], u, sd))
    ax1.clear()
    plt.plot(x,y,color="blue")

ani = animation.FuncAnimation(fig, animate, interval=50)
plt.show()