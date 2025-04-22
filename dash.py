import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import pysurveycto
import datetime
from plotnine import *
import re
import pytz
from wordcloud import WordCloud, STOPWORDS
import arabic_reshaper # to display arabic wordclouds
from bidi.algorithm import get_display # to display arabic wordclouds
from PIL import Image # for image display
import geopandas # to deal with kml files
import pyogrio # to deal with kml files and display polygons
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stat
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split


colors_lik={"1":"#DD3160","2":"#E39399","3":"#DAE1D9","4":"#94CEBA","5":"#00CEA4","98":"#AAAAAA","99":"#666666"}
colors_bin={"0":"#DD3160","1":"#00CEA4","98":"#AAAAAA","99":"#666666"}
colors_gap={"4":"#E76868","3":"#E76868","2":"#F59F5D","1":"#F59F5D","0":"#90BED7","-1":"#80B47E","-2":"#80B47E","-3":"#80B47E","-4":"#80B47E"}

gts_theme = (
    theme(
        text=element_text(family="Futura Bk BT"),
        legend_position="none",
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_background=element_blank(),
        axis_text_x=element_blank(),
        axis_ticks=element_blank(),
        axis_title_x=element_blank(),
        axis_text_y=element_blank(),
        axis_title_y=element_blank(),
    )
)

def round_to_100_percent(number_set, digit_after_decimal=0):
    """
        This function take a list of number and return a list of percentage, which represents the portion of each number in sum of all numbers
        Moreover, those percentages are adding up to 100%!!!
        Notice: the algorithm we are using here is 'Largest Remainder'
    """
    unround_numbers = [x / float(sum(number_set)) * 100 * 10 ** digit_after_decimal for x in number_set]
    decimal_part_with_index = sorted([(index, unround_numbers[index] % 1) for index in range(len(unround_numbers))], key=lambda y: y[1], reverse=True)
    remainder = 100 * 10 ** digit_after_decimal - sum([int(x) for x in unround_numbers])
    index = 0
    while remainder > 0:
        unround_numbers[decimal_part_with_index[index][0]] += 1
        remainder -= 1
        index = (index + 1) % len(number_set)
    return [int(x) / float(10 ** digit_after_decimal) for x in unround_numbers]

@st.cache_data
def likert_overall(data:pd.DataFrame,likerts:list) :
	likall=pd.DataFrame()
	for i in likerts:
		df = data[[i,"pweights","gender"]].groupby(i, as_index=False).aggregate({"pweights":"sum","gender":"count"})
		df["question"]=i
		if (df[df[i]<6]["pweights"].sum())/df["pweights"].sum()<=0.95:
			df_num=df[df[i]<6]
			n=df["gender"].sum()
			p=df["pweights"].sum()
			mean=df_num[i]*df_num["pweights"]/(df_num["pweights"].sum())
			df_num.loc[len(df_num)]=[98,p-df_num["pweights"].sum(),n-df_num["gender"].sum(),i]
			df_num["percent"] = round_to_100_percent(df_num["pweights"])
			df_num.columns=["score","pweights","n_cat","question","percent"]
			df_num["n"]=n
			df_num["mean"]=round(mean.sum(),2)
			likall = pd.concat([likall,df_num[["score","question","percent","n","mean"]]])
		else: 
			df=df[df[i]<6]
			n=df["gender"].sum()
			df["percent"] = round_to_100_percent(df["pweights"])
			df.columns=["score","pweights","n_cat","question","percent"]
			df["n"]=n
			mean=df["score"]*df["pweights"]/(df["pweights"].sum())
			df["mean"]=round(mean.sum(),2)
			likall = pd.concat([likall,df[["score","question","percent","n","mean"]]])
	likall["color"]=likall["score"].apply(lambda x: colors_lik[str(int(x))])
	likall["color"] = pd.Categorical(likall["color"], categories=list(colors_lik.values())[::-1], ordered=True)
	likall["pos"]=likall["score"].apply(lambda x: "left" if x<4 else "right")
	return likall

@st.cache_data
def binary_overall(data:pd.DataFrame,binarys:list) :
	binall=pd.DataFrame()
	for i in binarys:
		df = data[[i,"pweights","gender"]].groupby(i, as_index=False).aggregate({"pweights":"sum","gender":"count"})
		df["question"]=i
		if (df[df[i]<2]["pweights"].sum())/df["pweights"].sum()<=0.95:
			df_num=df[df[i]<2]
			n=df["gender"].sum()
			p=df["pweights"].sum()
			mean=df_num[i]*df_num["pweights"]/(df_num["pweights"].sum())
			df_num.loc[len(df_num)]=[98,p-df_num["pweights"].sum(),n-df_num["gender"].sum(),i]
			df_num["percent"] = round_to_100_percent(df_num["pweights"])
			df_num.columns=["score","pweights","n_cat","question","percent"]
			df_num["n"]=n
			df_num["mean"]=round(mean.sum(),2)
			binall = pd.concat([binall,df_num[["score","question","percent","n","mean"]]])
		else: 
			df=df[df[i]<2]
			n=df["gender"].sum()
			df["percent"] = round_to_100_percent(df["pweights"])
			df.columns=["score","pweights","n_cat","question","percent"]
			df["n"]=n
			mean=df["score"]*df["pweights"]/(df["pweights"].sum())
			df["mean"]=round(mean.sum(),2)
			binall = pd.concat([binall,df[["score","question","percent","n","mean"]]])
	binall["color"]=binall["score"].apply(lambda x: colors_bin[str(int(x))])
	binall["pos"]=binall["score"].apply(lambda x: "left" if x==0 else "right")
	return binall

@st.cache_data
def scqmcq_overall(data:pd.DataFrame,scqs:list,mcqs:list) :
	scqmcq=pd.DataFrame()
	for i in scqs:
		df = data[[i,"pweights"]]
		df.dropna(subset=i,inplace=True)
		n=len(df)
		df = data[[i,"pweights"]].groupby(i, as_index=False).aggregate({"pweights":"sum"})
		df["question"]=i
		#p=df["pweights"].sum()
		df["percent"] = round_to_100_percent(df["pweights"])
		df["n"]=n
		df.columns=["response","pweights","question","percent","n"]
		scqmcq = pd.concat([scqmcq,df[["response","question","percent","n"]]])
	for mcq in mcqs: # Pb with who_reports_payment_negative
		temp=pd.DataFrame(columns=["response","question","percent","n"])
		#st.write(data.columns)
		for feature in [i for i in data.columns if mcq+"__" in i[:(len(mcq)+2)]]:
			df=data[[mcq,feature,"pweights"]]
			df.dropna(subset=mcq,inplace=True)
			p=sum(df["pweights"])
			n=len(df[feature])
			df=df[df[feature]==1]
			if len(df)>0:
				response = " ".join(feature.split("__")[1:])
				pw=sum(df["pweights"])/p
				temp.loc[len(temp)]=[response,mcq,round(100*pw,0),n]
		scqmcq = pd.concat([scqmcq,temp[["response","question","percent","n"]]])

	return scqmcq

@st.cache_data
def scqmcq_bkdn(data:pd.DataFrame,scqs:list,mcqs:list,main_var:str,bkdn_var):
	bkdn_df = pd.DataFrame()
	if main_var in scqs:
		if bkdn_var not in mcqs:
			df_temp = data[[main_var,bkdn_var,"pweights"]].copy()
			for i in data[bkdn_var].unique():
				df=df_temp[df_temp[bkdn_var]==i][[main_var,"pweights"]]
				n=len(df)
				df = df.groupby(main_var, as_index=False).aggregate({"pweights":"sum"})
				df["question"]=main_var
				#p=df["pweights"].sum()
				df["percent"] = round_to_100_percent(df["pweights"])
				df["n"]=n
				df.columns=["response","pweights","question","percent","n"]
				df["bkdn_var"]=bkdn_var
				df["bkdn_value"]=i
				bkdn_df = pd.concat([bkdn_df,df[["response","question","bkdn_var","bkdn_value","percent","n"]]])
		else:
			for feature in [i for i in data.columns if bkdn_var+"__" in i[:(len(bkdn_var)+2)]]:
				df=data[data[feature]==1][[main_var,"pweights"]].copy()
				n=len(df)
				if len(df)>0:
					response_mcq = " ".join(feature.split("__")[1:])
					df = df.groupby(main_var, as_index=False).aggregate({"pweights":"sum"})
					df["question"]=main_var
					#p=df["pweights"].sum()
					df["percent"] = round_to_100_percent(df["pweights"])
					df["n"]=n
					df.columns=["response","pweights","question","percent","n"]
					df["bkdn_var"]=bkdn_var
					df["bkdn_value"]=response_mcq
					bkdn_df = pd.concat([bkdn_df,df[["response","question","bkdn_var","bkdn_value","percent","n"]]])
					#temp.loc[len(temp)]=[response,mcq,round(100*pw,0),n]

	else: # a Revoir n doit etre le nb de bkdn var pas de main_var
		bkdn_df=pd.DataFrame(columns=["response","question","bkdn_var","bkdn_value","percent","n"])
		if bkdn_var not in mcqs:
			for response in data[bkdn_var].unique():
				df=data[data[bkdn_var]==response]
				df=df[df[main_var]==df[main_var]]
				n=len(df)
				for feature in [i for i in data.columns if main_var+"__" in i[:(len(main_var)+2)]]:
					df_temp=df[[feature,"pweights"]]
					p=sum(df["pweights"])
					df_temp=df_temp[df_temp[feature]==1]
					if len(df_temp)>0:
						response_main = " ".join(feature.split("__")[1:])
						pw=sum(df_temp["pweights"])/p
						bkdn_df.loc[len(bkdn_df)]=[response_main,main_var,response,bkdn_var,round(100*pw,0),n]
		else: # remains this mcq mcq
			for response in [i for i in data.columns if bkdn_var+"__" in i[:(len(bkdn_var)+2)]]:
				response_bkdn = " ".join(response.split("__")[1:])
				df=data[data[response]==1][[i for i in data.columns if main_var+"__" in i[:(len(main_var)+2)]]+["pweights"]].copy()
				n=len(df)
				for feature in [i for i in data.columns if main_var+"__" in i[:(len(main_var)+2)]]:
					df_temp=df[[feature,"pweights"]]
					p=sum(df["pweights"])
					df_temp=df_temp[df_temp[feature]==1]
					if len(df_temp)>0:
						response_main = " ".join(feature.split("__")[1:])
						pw=sum(df_temp["pweights"])/p
						bkdn_df.loc[len(bkdn_df)]=[response_main,main_var,response_bkdn,bkdn_var,round(100*pw,0),n]


	return(bkdn_df)


# @st.cache_data
# def likerts_non_bkdn(data:pd.DataFrame,likerts:list,features:list) :
# 	likerts_bkdn=pd.DataFrame()
# 	for main_var in likerts:
# 		for second_var in features:
# 			df=data[["KEY",main_var,second_var,"pweights"]]
# 			df.dropna(subset=second_var,inplace=True)
# 			if second_var!=main_var:
# 				for resp in df[second_var].unique():
# 					temp=df[df[second_var]==resp]
# 					temp = temp[[main_var,"pweights","KEY"]].groupby(main_var, as_index=False).aggregate({"pweights":"sum","KEY":"count"})
# 					n=temp["KEY"].sum()
# 					temp["percent"] = round_to_100_percent(temp["pweights"])
# 					temp["n"]=n
# 					temp["response"]=second_var+" : "+str(resp)
# 					mean=temp[main_var]*temp["pweights"]/(temp["pweights"].sum())
# 					temp["mean"]=mean.sum()
# 					temp=temp[[main_var,"percent","n","response","mean"]]
# 					temp["summary"]="Mean = " + round(temp["mean"],1).astype(str) +" , n=" +temp["n"].astype(str)
# 					temp["main_var"]=main_var
# 					likerts_bkdn=pd.concat([likerts_bkdn,temp])
		
# 	likerts_bkdn["percent"]=likerts_bkdn["percent"].astype(int)
	
# 	return likerts_bkdn



def plot_likbin_bkdn(data:pd.DataFrame,main_var:str,second_var:str,mcqs:list,likerts:list,binarys:list,gaps:list) :
	bkdn=pd.DataFrame()
	
	if second_var not in mcqs:
		if "KEY" not in data.columns:
			data["KEY"]=data["id"]
		df=data[["KEY",main_var,second_var,"pweights"]]
		df.dropna(subset=second_var,inplace=True)
		for resp in df[second_var].unique():
			temp=df[df[second_var]==resp]
			temp = temp[[main_var,"pweights","KEY"]].groupby(main_var, as_index=False).aggregate({"pweights":"sum","KEY":"count"})
			n=temp["KEY"].sum()
			temp["percent"] = round_to_100_percent(temp["pweights"])
			temp["n"]=n
			temp["response"]=second_var+" : "+str(resp)
			temp2=temp.copy()
			temp2=temp2[temp2[main_var]<6]
			mean=temp2[main_var]*temp2["pweights"]/(temp2["pweights"].sum())
			temp["mean"]=mean.sum()
			temp=temp[[main_var,"percent","n","response","mean"]]
			temp["summary"]="Mean = " + round(temp["mean"],1).astype(str) +" , n=" +temp["n"].astype(str)
			bkdn=pd.concat([bkdn,temp])
		
	else:
		# Do the same for MCQs
		for feature in [i for i in data.columns if second_var+"__" in i]:
			df=data[[main_var,"pweights",feature]]
			df.dropna(subset=feature,inplace=True)
			response = " ".join(feature.split("__")[:2])
			n=df[feature].sum()
			temp=df[df[feature]==1]
			temp = temp[[main_var,"pweights"]].groupby(main_var, as_index=False).aggregate({"pweights":"sum"})
			temp["percent"] = round_to_100_percent(temp["pweights"])
			temp["response"]=second_var+" : "+str(response)
			temp2=temp.copy()
			temp2=temp2[temp2[main_var]<6]
			mean=temp2[main_var]*temp2["pweights"]/(temp2["pweights"].sum())
			temp["mean"]=mean.sum()
			temp["n"]=n
			temp=temp[[main_var,"percent","n","response","mean"]]
			temp["summary"]="Mean = " + round(temp["mean"],1).astype(str) +" , n=" +temp["n"].astype(str)
			bkdn=pd.concat([bkdn,temp])
	
	bkdn["percent"]=bkdn["percent"].astype(int)
	bkdn["result"]="Result in %"	
	bkdn=bkdn[bkdn["percent"]!=0]

	if main_var in likerts:
		bkdn["color"]=bkdn[main_var].apply(lambda x: colors_lik[str(int(x))])
		bkdn["color"] = pd.Categorical(bkdn["color"], categories=list(colors_lik.values())[::-1], ordered=True)
	elif main_var in binarys:
		bkdn["color"]=bkdn[main_var].apply(lambda x: colors_bin[str(int(x))])
		bkdn["color"] = pd.Categorical(bkdn["color"], categories=list(colors_bin.values())[::-1], ordered=True)
	else:
		bkdn["color"]=bkdn[main_var].apply(lambda x: colors_gap[str(int(x))])
		bkdn["color"] = pd.Categorical(bkdn["color"], categories=set(list(colors_gap.values())[::-1]), ordered=True)

	p = (
	ggplot(bkdn, aes(fill='color', y='percent', x='response', label='percent')) +
    geom_bar(position='stack', stat='identity', color='white', width=0.35, size=0.35) +
	coord_flip() +
    geom_text(aes(label='percent', x='response'), size=10, family='Futura Bk BT', position=position_stack(vjust=0.5)) +
	geom_text(aes(y=15, label='response', x='response'),
          color='black', family='Futura Bk BT', position=position_nudge(x=0.25), size=10) +
    geom_text(aes(x='response', y=85, label="summary"),
          color='black', family='Futura Bk BT',position=position_nudge(x=0.25), size=10) +
    geom_text(aes(x=0.1, y=95, label='result'),
          color='black', family='Futura Bk BT', position=position_nudge(x=0.4), size=10) +
    scale_fill_identity() +
    gts_theme 
	)
	return p

		

	return bkdn


def plot_div(df:pd.DataFrame,likert=True):
	"""
	Plot divergent bar chart
	"""
	categories=["#DAE1D9","#E39399","#DD3160","#94CEBA","#00CEA4"][::-1] if likert else ["#DD3160","#00CEA4"]

	df["percent"]=df["percent"].astype(int)
	df["percent"] = df.apply(lambda row: -row["percent"] if row["pos"]=="left" else row["percent"],axis=1)
	df["per"]=abs(df["percent"])
	df=df.sort_values("mean")
	df=df[df["score"]<6]
	df['question'] = pd.Categorical(df['question'], categories=df.question.unique(), ordered=True)
	df['color'] = pd.Categorical(df['color'], categories=[i for i in categories if i != "#AAAAAA"], ordered=True)
	
	p = (
    ggplot(df, aes(fill='color', y='percent', x='question')) +
    geom_bar(position='stack', stat='identity', color='white', width=0.525, size=0.1) +
    scale_fill_identity() +
    aes(y='percent') +
    #ylab(f"n={df['n'].max()}, Results in %") +
    scale_y_continuous(breaks=range(df['percent'].min(), df['percent'].max() + 1, 25)) +
    geom_hline(yintercept=0, color='black') +
    geom_text(aes(label='per'),
              size=12, family='Futura Bk BT', position=position_stack(vjust=0.5)) +
    coord_flip() +
    theme(
        text=element_text(size=15, family='Futura Bk BT'),
        strip_background=element_blank(),
        strip_text_x=element_blank(),
        panel_grid_major_x=element_line(color='lightgray', linetype='solid'),  # Updated this line
        panel_background=element_rect(fill='white', colour='#B0B0B0'),  # Updated this line
        axis_ticks_major_x=element_blank(),
        axis_text_x=element_blank(),
        axis_title_x=element_text(hjust=1, vjust=0, size=10),
        axis_title_y=element_blank(),
		figure_size=(16, 0.6*len(df.question.unique())+1)
    )
	)

	return p

def mcq_plot(df:pd.DataFrame,height=1):
	df.n="n= "+df.n.astype(str)
	df["percent"]=df.percent.astype(int)
	df.sort_values("percent",ascending=True,inplace=True)
	order=df.response.unique().tolist()
	df['response'] = pd.Categorical(df['response'], categories=order, ordered=True)

	plot = (
		ggplot(df, aes(y='percent', x='response', label='percent')) +
		geom_bar(stat="identity", color="white", width=0.45) +
		coord_flip() +
		geom_text(size=13*height, family="Futura Bk BT", position=position_stack(vjust=0.5),color="white") +
		geom_text(aes(y=0, label='response', x='response'),#vjust=0,
				color="black", family="Futura Bk BT",
				nudge_x=0.4, size=12*height,ha="left") +
		geom_text(aes(x=0.15, y=df.percent.max(), label='res_label'),
				color="black", family="Futura Bk BT",
				position=position_nudge(x=0.4), size=12*height,ha="center") +
		geom_text(aes(x=len(df), y=df.percent.max(), label="n"),
				color="black", family="Futura Bk BT",
				position=position_nudge(x=0.4), size=12*height,ha="center") +
		scale_fill_identity() +
		theme(
			figure_size=(16, 0.6*len(df.response.unique())*height)			    # Rotate x-axis labels
		) +
		gts_theme
		)
	return plot


def plot_mcq(df:pd.DataFrame,cols:int,height:int,width=16):
	"""
	Creates the likert MCQ a tile plots with colors per response
	
    Keyword arguments:
    df -- the mcq dataframe to display (potentially filtered by enum and time)
	cols -- number of columns
	height and width: the size of the plot    
	"""
	if len(df)==0: 
		return ggplot()
	p_likert = (ggplot(df, aes(x='counter_enum', y='question'))
            + geom_tile(aes(fill='color', label='value'))
            + facet_wrap('~enumerator', ncol=cols)
            + scale_fill_identity()
            + theme(axis_text_x=element_blank(), axis_ticks=element_blank(), legend_position="bottom",figure_size=(width, height))
            + xlab("")
            + ylab("")
            + labs(title="Likert graphs"))
	return p_likert



def plot_likert(df:pd.DataFrame,cols:int,height:int,width=16):
	"""
	Creates the likert plot a tile plots with colors per response
	
    Keyword arguments:
    df -- the likerts dataframe to display (potentially filtered by enum and time)
	cols -- number of columns
	height and width: the size of the plot    
	"""
	
	df["title"]=df.enumerator + "- std_alt: " + round(df.std_alternative,2).astype(str)
	#df["color_tile"]=df.std_alternative.apply(lambda x: "red" if x<1 else ("lightgreen" if x<1.2 else "green")) trial to change the backgorund of the tiles but looks not possible with plotnine
	# maybe we could add a rectangle behid the tiles in a certain color based on the value of std_alt
	
	# colors = (
    # 		df
    # 		.groupby('enumerator')
    # 		.agg(color_tile=('color_tile', 'first'))
    # 		.reset_index()
	# 		)
	
	# colors=colors.color_tile.to_list()

	if len(df)==0: 
		return ggplot()
	p_likert = (ggplot(df, aes(x='counter_enum', y='question'))
            + geom_tile(aes(fill='color', label='value'))
            + facet_wrap('~title', ncol=cols)
            + scale_fill_identity()
            + theme(axis_text_x=element_blank(), axis_ticks=element_blank(), legend_position="bottom",figure_size=(width, height))
            + xlab("")
            + ylab("")
            + labs(title="Likert graphs"))
	return p_likert

def display_gap(data:pd.DataFrame):
	"""
	Creates the ECT plot for all the vriables including percetion or expect in the name
	
    Keyword arguments:
    data -- the data to display    
	"""
	
	df=data[["today"]+[i for i in data if "perception_" in i or "expect" in i]].copy() #extract variables
	for i in [i for i in df if "perception_" in i or "expect" in i]: # convert values to floats
		df[i]=df[i].apply(lambda x: np.nan if x=="" else x).astype(float)
	df_long=pd.melt(df,id_vars="today",var_name="name",value_name="value").dropna(subset="value") # create long df
	df_long["category"]=df_long["name"].apply(lambda x:x.split("_")[0]) # extract perception or expectation
	df_long["name"]=df_long["name"].apply(lambda x:"_".join(x.split("_")[1:])) # extract name
	df_long=df_long[df_long["value"]<90] # remove 98 and 99
	gap_df=df_long.groupby(["category","name"])["value"].agg(mean='mean').reset_index() # extract mean values
	gap_plot = ( # creates the plot
    			ggplot(gap_df, aes(y='mean', x='category')) +
    			geom_bar(stat="identity") +
				facet_wrap('~name', ncol=5, scales="free_x") +
    			coord_cartesian(ylim=(1, 5)) +
    			theme(figure_size=(10, 6))  # Adjust figure size as needed
				)
	return gap_plot




def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def listing(var,xls,mcqs_choices,mcqs,scqs,binarys,demographics,likerts):
	if var in mcqs_choices:
		var=[k for k in mcqs if k in var[:len(k)]][0]	
	relevant=xls[xls["name"]==var]["relevant"].iloc[0] if var in xls["name"] else []
	toremove=[var for var in scqs+mcqs+likerts+binarys+demographics if var in relevant] if relevant==relevant else []
	return toremove+[k for k in xls[xls["relevant"].apply(lambda x:False if x!=x or var not in x else True)]["name"].tolist() if "text" not in k]

def to_remove(var,xls,mcqs_choices,mcqs,scqs,binarys,demographics,likerts):
	toremove=listing(var,xls,mcqs_choices,mcqs,scqs,binarys,demographics,likerts)
	temp=[i for i in toremove]
	remove2=[]
	while len(temp)!=0:
		for k in temp:
			remove2 += listing(k,xls,mcqs_choices,mcqs,scqs,binarys,demographics,likerts)
		temp=set([i for i in remove2 if i not in toremove and i!=k])
		toremove += [i for i in temp if i !=var]
		remove2=[]
	return toremove

@st.cache_data
def corr_matrix(data,regression):
	dummy = pd.get_dummies(data[regression+["pweights"]], columns=regression, drop_first=True)
	cols=[i for i in dummy.columns if i != "pweights"]
	corr_mat=pd.DataFrame(columns=cols,index=cols)
	for i in cols:
		dummy[i]=dummy[i].astype(float)
	for i in cols:
		for j in cols:
			if i==j:
				corr_mat.loc[i][j]=1
			else:
				corr_mat.loc[i][j]=corr(dummy[i],dummy[j],dummy["pweights"])
	mask = np.triu(np.ones_like(corr_mat))[1:,:-1]
	heatmap = sns.heatmap(corr_mat.iloc[1:][corr_mat.columns.tolist()[:-1]].astype(float).round(2),mask=mask, vmin=-1, vmax=1, annot=True,cmap='BrBG')
	heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
	return heatmap

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df


@st.cache_data
def regression(df,variable):

	if "age_group_51+" in df.columns: # For Nigeria, need to find a more systematic way with "re"
		df=df.rename(columns={"age_group_51+":"age_group_51","age_group_36-50":"age_group_36_50"})

	df=df[df[variable]<6]

	glm = smf.glm(
					variable + "~ "+" + ".join([i for i in df.columns if i!=variable and i!="pweights"]),
					data=df,
					family=sm.families.Gaussian(),
					freq_weights=np.asarray(df["pweights"]*len(df)),
					)
	res_f = glm.fit()
	summ=res_f.summary()
	st.write(summ)
	#summ2=res_f.summary2()
	summ_table=results_summary_to_dataframe(res_f)
	prediction=res_f.predict(df[[i for i in df.columns if i!=variable and i!="pweights"]])
	rsquare=[i for i in summ.tables[0].as_csv().split(" ") if "CS" in i][0].split(",")[1]
	
	return rsquare,summ_table,pd.DataFrame({"Real":df[variable],"Predicted":prediction})

@st.cache_data
def lgbm(df,variable,to_remove,categoricals,top=6,training=False):

	to_drop=[i for i in to_remove if i not in categoricals]
	drop_smcq=[i for i in to_remove if i in categoricals]
	for c in drop_smcq:
		to_drop += [i for i in df if c in i[:len(c)]]

	perc=[i for i in df if "perception_" in i]
	exp=[i for i in df if "expect_" in i]
	if variable in perc+exp:
		to_drop+=[i for i in perc+exp if i not in to_drop]

	to_drop=[i for i in to_drop if i!= variable]+["pweights"]
	df.drop([i for i in to_drop if i in df],axis=1,inplace=True)

	binaire=[i for i in df.columns if len(df[i].unique())==2 and set(df[i].unique()).issubset(set([0,1]))]

	model=lgb.LGBMRegressor()
	y=df[variable].copy()
	X=df.drop(variable,axis=1).copy()
	
	if training:
		X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2)
		model.fit(X_train,y_train)
		y_pred=model.predict(X_test)
		explainer = shap.TreeExplainer(model)
		svalues=explainer.shap_values(X,y)
		feature_importance = pd.DataFrame(list(zip(X.columns, sum(abs(svalues)))), columns=['col_name','feature_importance_vals'])
		feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
		#extract top 6:
		tops=feature_importance["col_name"].tolist()[:top]
		return svalues, X, pd.DataFrame({"Real":y_test,"Predicted":y_pred}),tops  # Look how to display this plot on streamlit

	else:
		model.fit(X,y)
		y_pred=model.predict(X)
		explainer = shap.TreeExplainer(model)
		svalues=explainer.shap_values(X,y)
		feature_importance = pd.DataFrame(list(zip(X.columns, sum(abs(svalues)))), columns=['col_name','feature_importance_vals'])
		feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
		#extract top 6:
		tops=feature_importance["col_name"].tolist()[:top]
		return svalues, X, pd.DataFrame({"Real":y,"Predicted":y_pred}),tops  # Look how to display this plot on streamlit
	


@st.cache_data
def extract_data(data_xlsx):
	xls=pd.read_excel(data_xlsx,sheet_name="survey")
	xls=xls[xls.name==xls.name]
	xls=xls[xls.name.apply(lambda x: False if x[:6]=="start_" else True)]
	xls=xls[xls.name.apply(lambda x: False if x[:4]=="end_" else True)]
	xls=xls[xls.name.apply(lambda x: False if x[:5]=="time_" else True)]

	xls["name"] = xls["name"].apply(lambda x: "_".join(x.split(".")))

	data=pd.read_excel(data_xlsx,sheet_name="data_protected")
	choices=pd.read_excel(data_xlsx,sheet_name="choices")
	if "pweights" not in data.columns:
		data["pweights"]=(data.weight)/sum(data.weight)
	data.columns = ["_".join(x.split(".")) for x in data.columns]
	data.columns = ["__".join(x.split("/")) for x in data.columns]
	data.columns = ["_".join(x.split(" ")) for x in data.columns]

	for i in data.columns:
		if data[i].dtype == "object":
			data[i] = data[i].apply(lambda x: x if x!=x else "_".join(x.split(" ")))
			data[i] = data[i].apply(lambda x: x if x!=x else "_".join(x.split(",")))
			data[i] = data[i].apply(lambda x: x if x!=x else "_".join(x.split("'")))
			data[i] = data[i].apply(lambda x: x if x!=x else "_and_more".join(x.split("+")))			
		if "categories" in i:
			data[i] = data[i].fillna(0)
			data[i] = data[i].astype(int)

	#data["settlement_type"] = data["settlement_type"].apply(lambda x: "urban_type" if "Urban" in x else x)

	def func(x):
		if "(registered_at" in x:
			return "unemployed_registered"
		elif "(unregistered" in x:
			return "unemployed_unregistered"
		else:
			return x

	#data["employment"] = data["employment"].apply(func)


	demographics = xls[xls["question_section"]=="demographics"]["name"].tolist()
	meta=xls[xls["question_section"]=="metadata"]["name"].tolist()
	main=xls[xls["question_section"]=="main_questions"]["name"].tolist()
	likerts = xls[xls["type"].isin([i for i in xls["type"].dropna() if "likert" in i])]["name"].tolist()
	binarys = xls[xls["type"].isin([i for i in xls["type"].dropna() if "select_one yesno" in i])]["name"].tolist()
	scqs=xls[xls["type"].isin([i for i in xls["type"].dropna() if "select_one" in i])]["name"].tolist()
	scqs=[i for i in scqs if i not in binarys+likerts and i in main]
	mcqs=xls[xls["type"].isin([i for i in xls["type"].dropna() if "multiple" in i])]["name"].tolist()
	mcqs=[i for i in mcqs if "survey" not in i]
	mcqs_choices=[]
	for quest in mcqs:
		mcqs_choices += [i for i in data.columns if quest+"__"==i[:len(quest)+2] and "text" not in i]
	demographics = [i for i in demographics if "diff" not in i and i not in ["age","loc_3"] and "text" not in i]
	if "disabled" in data.columns and "disabled" not in demographics:
		demographics += ["disabled"]
	if "age_group" in data.columns and "age_group" not in demographics:
		demographics += ["age_group"]
	
	
	demographics=[i for i in demographics if len(data[i].unique())<len(data)]
	binarys=[i for i in binarys if i in main]
	gaps=[i for i in data.columns if i[:3]=="gap"]	

	return xls,data,choices,demographics,meta,main,likerts,binarys,scqs,mcqs,demographics,gaps,mcqs_choices
