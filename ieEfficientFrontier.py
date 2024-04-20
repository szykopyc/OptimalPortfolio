#-- all of the required libraries
import glob
import pandas as pd
import numpy as np 
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from operator import itemgetter
import sys
import os

#-- List files to read
source_files = glob.glob('CSV_Files/*.csv')
print(source_files)
input()

#-- percentage change array
source_daily_perc_change = []

#-- function to enable data to go straight to a log file

def logdata():
#-- outputs all the data into a folder called OutputFiles
	output_file_name = "OutputFiles/output-#.txt"
	output_version = 1
	while os.path.isfile(output_file_name.replace("#", str(output_version))):
		output_version += 1
	
	output_file_name = output_file_name.replace("#", str(output_version))
	sys.stdout = open(output_file_name,'wt')

logdata() #- comment this function call out if you don't want to log the data

#-- goes through the files, calculates percentage change and appends to source array
for i in source_files:
	f = pd.read_csv(i)
	f_close = f[["Close"]]
	f_close = f_close.to_numpy()
	f_close = f_close.flatten()
	f_perc_change = np.diff(f_close) / f_close[1:] * 100
	source_daily_perc_change.append(f_perc_change)

####### the below arrays are a very tedious way of adding/removing assets to use
####### but to keep the code simple-ish i have left it this way

#-- extracts symbol specific from source array in format:
#-- [ticker, %Δ array, daily mean %Δ, yearly return, variance]
XLEP = ["XLEP",source_daily_perc_change[0]]
ESIE = ["ESIE",source_daily_perc_change[1]]
VFEG = ["VFEG",source_daily_perc_change[2]]
XLKQ = ["XLKQ",source_daily_perc_change[3]]
UIFS = ["UIFS",source_daily_perc_change[4]]
IITU = ["IITU",source_daily_perc_change[5]]
VWRL = ["VWRL",source_daily_perc_change[6]]
EEIP = ["EEIP",source_daily_perc_change[7]]
V3AB = ["V3AB",source_daily_perc_change[8]]
VUSA = ["VUSA",source_daily_perc_change[9]]
ICDU = ["ICDU",source_daily_perc_change[10]]
HMWO = ["HMWO",source_daily_perc_change[11]]
EQQQ = ["EQQQ",source_daily_perc_change[12]]
LGUG = ["LGUG",source_daily_perc_change[13]]
CB5_ = ["CB5",source_daily_perc_change[14]]
SWDA = ["SWDA",source_daily_perc_change[15]]


#-- calculates the daily % change mean in terms of %
XLEP.append(XLEP[1].mean())
ESIE.append(ESIE[1].mean())
VFEG.append(VFEG[1].mean())
XLKQ.append(XLKQ[1].mean())
UIFS.append(UIFS[1].mean())
IITU.append(IITU[1].mean())
VWRL.append(VWRL[1].mean())
EEIP.append(EEIP[1].mean())
V3AB.append(V3AB[1].mean())
VUSA.append(VUSA[1].mean())
ICDU.append(ICDU[1].mean())
HMWO.append(HMWO[1].mean())
EQQQ.append(EQQQ[1].mean())
LGUG.append(LGUG[1].mean())
CB5_.append(CB5_[1].mean())
SWDA.append(SWDA[1].mean())

#-- calculates the yearly %Δ in terms of %, e.g. 10=10%
XLEP.append(XLEP[2]*252)
ESIE.append(ESIE[2]*252)
VFEG.append(VFEG[2]*252)
XLKQ.append(XLKQ[2]*252)
UIFS.append(UIFS[2]*252)
IITU.append(IITU[2]*252)
VWRL.append(VWRL[2]*252)
EEIP.append(EEIP[2]*252)
V3AB.append(V3AB[2]*252)
VUSA.append(VUSA[2]*252)
ICDU.append(ICDU[2]*252)
HMWO.append(HMWO[2]*252)
EQQQ.append(EQQQ[2]*252)
LGUG.append(LGUG[2]*252)
CB5_.append(CB5_[2]*252)
SWDA.append(SWDA[2]*252)


#-- calculates covariance of each ETF in real value, for example 1.25 = 125%
XLEP.append(np.var(XLEP[1]))
ESIE.append(np.var(ESIE[1]))
VFEG.append(np.var(VFEG[1]))
XLKQ.append(np.var(XLKQ[1]))
UIFS.append(np.var(UIFS[1]))
IITU.append(np.var(IITU[1]))
VWRL.append(np.var(VWRL[1]))
EEIP.append(np.var(EEIP[1]))
V3AB.append(np.var(V3AB[1]))
VUSA.append(np.var(VUSA[1]))
ICDU.append(np.var(ICDU[1]))
HMWO.append(np.var(HMWO[1]))
EQQQ.append(np.var(EQQQ[1]))
LGUG.append(np.var(LGUG[1]))
CB5_.append(np.var(CB5_[1]))
SWDA.append(np.var(SWDA[1]))

#-- array off all sub-arrays of the tickers
TICKER_ARRAY = [XLEP,ESIE,VFEG,XLKQ,UIFS,IITU,VWRL,EEIP,V3AB,VUSA,ICDU,HMWO,EQQQ,LGUG,CB5_,SWDA]
TICKER_ARRAY = np.asarray(TICKER_ARRAY, dtype=object)

#-- now we start building the portfolio generator

#-- number of portfolios to generate and how many assets in each
n_portfolios = 20000
n_assets = 8

#-- empty list to store mean-variance points
mean_variance_pairs = []

#-- seed for reproducing the results
np.random.seed(3)

#-- the portfolio generator
print(f"Generating {n_portfolios} portfolios made of {n_assets} assets...\n")
for i in tqdm(range(n_portfolios)):

	#-- randomly choose which assets go into the portfolio
	assets = TICKER_ARRAY[np.random.choice(TICKER_ARRAY.shape[0], n_assets, replace=False)]
	
	#-- randomly choose weights and ensure they total to 1
	weights = np.random.rand(n_assets)
	weights = weights/sum(weights)

	#-- now compute portfolio return and variance
	portfolio_E_return = 0
	portfolio_E_variance = 0
	portfolio_assets_weights = []

	#-- appends the assets and weights to an array
	for i in range(len(assets)):
		portfolio_assets_weights.append(assets[i][0])
	
	#-- rounds the weight to 2 decimal places for integer percentages later on	
	weights_rounded = []
	for i in weights:
		i = round(i,2)
		weights_rounded.append(i)
		
	portfolio_assets_weights.append(weights_rounded)

	#-- calculates the portfolio return and portfolio variance
	for i in range(len(assets)):
		portfolio_E_return += weights[i] * assets[i][3]

		for j in range(len(assets)):
			portfolio_E_variance += (weights[i]**2) * (assets[i][4])+(weights[j]**2) * (assets[j][4]) + (2*weights[i]*weights[j]*np.cov([i,j]))

	#- append to list
	mean_variance_pairs.append([portfolio_E_return,portfolio_E_variance, portfolio_assets_weights])


#-- Plot the risk vs. return of randomly generated portfolios
#-- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs, dtype=object)

#-- modify the risk-free rate here
#-- currently using the average 2019-2024 RFR
risk_free_rate=2.198

#-- draws the graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], text=mean_variance_pairs[:,2].tolist(), 
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                  showscale=True, 
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="Aggrnyl", #- RdBu, Aggrnyl, Purpor - examples of colour schemes for the scale
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ), 
                      mode='markers'))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility) %'),
                  yaxis=dict(title='Annualised Return in %'),
                  title='Sample of random portfolios generated from a list of 16 ETFs',
                  width=847, #- change the width and height of the graph here, my current setting
                  height=433) #- for fullscreen is 1694 by 865
fig.update_xaxes(range=[0.18, 0.32])
fig.update_yaxes(range=[0.02,0.27])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))

#-- arrays of all sharpe ratios, their corresponding tickers, return and volatility
sharpe_list = []
corresponding_tickers = []
corresponding_return = []
corresponding_volatility = []

#-- appends to the arrays mentioned previously
sharpe_ratio = (mean_variance_pairs[:,0] - risk_free_rate)/(mean_variance_pairs[:,1]**0.5)
sharpe_list.append(sharpe_ratio)
corresponding_tickers.append(mean_variance_pairs[:,2])
corresponding_return.append(mean_variance_pairs[:,0])
corresponding_volatility.append(mean_variance_pairs[:,1]**0.5)

#-- sorts all the arrays by the value of the sharpe ratio from smallest to largest
def sort_sharpe_corresp(sharpe_l,corresp,p_return,p_volatility):
	sharpe_and_ticker = []
	sharpe_l = np.array(sharpe_l).flatten()
	corresp = np.array(corresp).flatten()
	p_return = np.array(p_return).flatten()
	p_volatility = np.array(p_volatility).flatten()

	#-- stacks all the corresponding arrays for the sharpe ratio together by column
	sharpe_and_ticker = np.stack((sharpe_l, corresp, p_return, p_volatility),axis=1)

	sorted_list = sorted(sharpe_and_ticker, key=itemgetter(0))

	return sorted_list #- returns the sorted list

#-- function for showing the extremes of the data
#-- takes in the sorted list, the top/bottom data to show, and the max/min weight filter
def show_extreme_portfolios(srted,n_top_min,filtr,min_filtr):

	srted = srted

	#-- the counters
	counter_top = 0
	counter_bottom = 0
	asset_ctr = n_assets

	#-- the filtered list
	filtered = []

	#-- weight filter for loop, filters through sorted list using a set weight

	for i in srted:
		to_append = None
		for j in i:
			if type(j) is list:
				weights_fetcher = j[asset_ctr]
				to_append = all(min_filtr <= k <= filtr for k in weights_fetcher)

		if to_append == True:
			filtered.append(i)
		else:
			pass

	#-- final filter for the highest and lowest sharpe ratio, uses the n of how many pairs to show
	top_n_sharpes = filtered[-n_top_min:]
	top_n_sharpes.reverse()
	bot_n_sharpes = filtered[:n_top_min]

	print(f"List of the top/bottom {n_top_min} portfolios ranked\nby Sharpe Ratio with maximum weight filter of {filtr*100}%\nand minimum weight filter of {min_filtr*100}%")

	#-- prints the top n sharpe ratio pairs, but doesn't print to console - prints to the output file
	print(f"\nTop {n_top_min} Sharpe Ratios:\n")
	for i in top_n_sharpes:
		counter_top+=1
		print(str(counter_top) + ". Sharpe Ratio = "+ str(round(i[0],2)))
		print("Annualised return = "+str(round(i[2],2))+"%")
		print("Annualised risk (Volatility) = "+str(round(i[3],2))+"%")

		if type(i[1]) is list:
			print(*i[1][:asset_ctr], sep=", ")
			weights_fetcher = i[1][-1:]
			tmp = []
			for i in weights_fetcher[0]:
				to_percent = round(i*100,0)
				to_percent = str(to_percent)
				to_percent = to_percent[:-2]+"%"
				tmp.append(to_percent)
			weights_fetcher_string = tmp
			print(*weights_fetcher_string,sep=',  ')
		print("")

	#-- prints the bottom n sharpe ratio pairs, but doesn't print to console - prints to the output file
	print(f"Bottom {n_top_min} Sharpe Ratios:\n")
	for i in bot_n_sharpes:
		counter_bottom+=1
		print(str(counter_bottom) + ". Sharpe Ratio = "+ str(round(i[0],2)))
		print("Annualised return = "+str(round(i[2],2))+"%")
		print("Annualised risk (Volatility) = "+str(round(i[3],2))+"%")
		#-------for j in i:

		if type(i[1]) is list:
			print(*i[1][:asset_ctr], sep=", ")
			weights_fetcher = i[1][-1:]
			tmp = []
			for i in weights_fetcher[0]:
				to_percent = round(i*100,0)
				to_percent = str(to_percent)
				to_percent = to_percent[:-2]+"%"
				tmp.append(to_percent)
			weights_fetcher_string = tmp
			print(*weights_fetcher_string,sep=',  ')
		print("")

#-- calls the function which sorts sharpe ratio pairs
sorted_sharpe = sort_sharpe_corresp(sharpe_list,corresponding_tickers, corresponding_return,corresponding_volatility)


#-- Shows the extreme portfolios, with these arguments - sorted sharpe list, n top/bottom, max weight, min weight
show_extreme_portfolios(sorted_sharpe,5,0.25,0.05)

#-- shows the graph
fig.show()