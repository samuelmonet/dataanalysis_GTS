from xmlrpc.client import boolean
import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go # for plots if we want more interactive plots
import plotly.express as px # for plots of maps
from plotly.subplots import make_subplots # for displaying multiple plots in one call
import matplotlib.pyplot as plt
from plotnine import * # library to use ggplot coding
from collections import Counter
from PIL import Image # for image display
from streamlit_option_menu import option_menu # for sidebar
import re # regex package
import dash # the package with the designed functions
import math
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stat
import lightgbm as lgb
import shap
import streamlit.components.v1 as components
import io
from sklearn.metrics import r2_score

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():	
	
	
	with st.sidebar:		
		data_xlsx = st.file_uploader("Upload dataset from vault", type=".xlsx")
		
		topic = option_menu(None, ["Data","Likerts & Binaries","SCQ & MCQ", "Cross_data"],
							 icons=["table","bar-chart-steps","bar-chart-line", "robot"], # https://icons.getbootstrap.com/
							 menu_icon="app-indicator", default_index=0,
							 ) # set up the choice bar in the sidebar
	
	xls,data,choices,demographics,meta,main,likerts,binarys,scqs,mcqs,demographics,gaps,mcqs_choices=dash.extract_data(data_xlsx)

	try:
		xls,data,choices,demographics,meta,main,likerts,binarys,scqs,mcqs,demographics,gaps,mcqs_choices=dash.extract_data(data_xlsx)
		
	except:
		st.header([":red[Upload a dataset]"])
	
	if topic=="Data":
		try:
			st.title("Data")
			st.dataframe(data)
			st.title("XlS form")
			st.dataframe(xls)
			st.title("Choices")
			st.dataframe(choices)

		except:
			st.write("Drag and drop a dataset in the sidebar")


	elif topic=="Likerts & Binaries":

		tab1, tab2, tab3,tab4, tab5,tab6 = st.tabs(["Divergent Chart Likerts","Divergent Chart Binary", "Gaps", "Likerts Breakdowns", "Binary Breakdowns","Gaps breakdowns"])

		with tab1:
			likerts_overall=dash.likert_overall(data,likerts)
			st.pyplot(ggplot.draw(dash.plot_div(likerts_overall)))

		with tab2:
			bin_overall=dash.binary_overall(data,binarys)
			st.pyplot(ggplot.draw(dash.plot_div(bin_overall,likert=False)))
			
		with tab3:
			if len(gaps)>0:
				likerts_overall=dash.likert_overall(data,likerts)
				perceptions=[re.sub("gap","perception",i) for i in gaps]
				expectations=[re.sub("gap","expect",i) for i in gaps]
							
				ECT_mean = likerts_overall[likerts_overall['question'].isin(perceptions+expectations)].loc[:, ['question', 'mean']]
				ECT_mean.drop_duplicates(inplace=True)
				ECT_mean[['group', 'category']] = ECT_mean['question'].str.split('_', n=1, expand=True)
				# Use pivot_table instead of pivot_wider
				ECT_mean = ECT_mean.pivot_table(index=['category'], columns='group', values='mean', aggfunc='first')
				# Reset index to make 'category' a regular column
				ECT_mean.reset_index(inplace=True)

				p = (
				ggplot() +
				geom_segment(ECT_mean,
							aes(y='expect', yend='perception',
								x='category', xend='category'),
							alpha=0.7, color="#B0B0B0", size=1) +
				scale_y_continuous(breaks=[1, 3, 5],
								labels=["Strongly disagree", "Neutral", "Strongly agree"],
								limits=[1, 5]) +
				labs(x=None, y=None, color=None) +
				theme(
					text=element_text(family="Futura Bk BT"),
					axis_line_y=element_line(arrow=True, size=0.3, ends="both"),
				) +
				geom_point(ECT_mean.melt(id_vars='category', var_name='group'),
						aes(x='category', y='value', color='group'), size=5) +
				theme_bw()+
				theme(axis_text_x=element_text(angle=45, hjust=1))
				)				
				st.pyplot(ggplot.draw(p))
			else:
				st.write("No gap measured in this survey")
			
		with tab4:
			col1,col2=st.columns([1,1])
			main_var=col1.selectbox("Select the likert variable",likerts)
			dem=col2.multiselect("1/ Select breakdown variables in demog",[i for i in demographics if i!=main_var])
			likbin=col2.multiselect("1/ Select breakdown variables in likert and binaries",[i for i in likerts+binarys if i!=main_var])
			smcq=col2.multiselect("1/ Select breakdown variables in MCQ and SCQ",[i for i in mcqs+scqs if i!=main_var])
			second_vars=dem+likbin+smcq
			if col1.button("1/ Display") and len(second_vars)>0:
				for i in second_vars:
					st.subheader(i+ " vs " +main_var)
					st.pyplot(ggplot.draw(dash.plot_likbin_bkdn(data,main_var,i,mcqs,likerts,binarys,gaps)))

		with tab5:
			col1,col2=st.columns([1,1])
			main_var=col1.selectbox("Select the binary variable",binarys)
			dem=col2.multiselect("2/ Select breakdown variables in demog",[i for i in demographics if i!=main_var])
			likbin=col2.multiselect("2/ Select breakdown variables in likert and binaries",[i for i in likerts+binarys if i!=main_var])
			smcq=col2.multiselect("2/ Select breakdown variables in MCQ and SCQ",[i for i in mcqs+scqs if i!=main_var])
			second_vars=dem+likbin+smcq
			if col1.button("2/ Display") and len(second_vars)>0:
				for i in second_vars:
					st.subheader(i+ " vs " +main_var)
					st.pyplot(ggplot.draw(dash.plot_likbin_bkdn(data,main_var,i,mcqs,likerts,binarys,gaps)))

		with tab6:
			if len(gaps)>0:
				col1,col2=st.columns([1,1])
				main_var=col1.selectbox("Select the gap variable",gaps)
				dem=col2.multiselect("3/ Select breakdown variables in demog",[i for i in demographics if i!=main_var])
				likbin=col2.multiselect("3/ Select breakdown variables in likert and binaries",[i for i in likerts+binarys if i!=main_var])
				smcq=col2.multiselect("3/ Select breakdown variables in MCQ and SCQ",[i for i in mcqs+scqs if i!=main_var])
				second_vars=dem+likbin+smcq
				if col1.button("3/ Display") and len(second_vars)>0:
					for i in second_vars:
						st.subheader(i+ " vs " +main_var)
						st.pyplot(ggplot.draw(dash.plot_likbin_bkdn(data,main_var,i,mcqs,likerts,binarys,gaps)))
			else:
				st.write("No gap measured in this survey")

	elif topic=="SCQ & MCQ":
		#st.write(data["who_reports_payment_negative"].unique())
		scqmcq_overall=dash.scqmcq_overall(data,scqs,mcqs)
		main_var=st.selectbox("Select the variable you want to display",scqs+mcqs)
		df = scqmcq_overall[scqmcq_overall["question"]==main_var]
		df["res_label"]="Results in %"
		
		plot=dash.mcq_plot(df)
		
		st.pyplot(ggplot.draw(plot))
		#st.pyplot(ggplot.draw(dash.plot_mcq(scqmcq_overall[scqmcq_overall["question"]==main_var])))

		col1,col2,col3=st.columns([1,1,1])
		bkdn=col1.checkbox("Display breakdowns")
		if bkdn:
			second_var=col2.selectbox("Select the breakdown variable you want to display",[i for i in demographics+binarys+scqs+mcqs+likerts if i!=main_var])
			data_bkdn = dash.scqmcq_bkdn(data,scqs,mcqs,main_var,second_var)
			data_bkdn["res_label"]="Results in %"
			
			k=0
			for i in data_bkdn["bkdn_var"].unique():
				k+=1
				
				if k%3==1:
					col1,col2,col3=st.columns([1,1,1])
					col1.header(i)
					#col1.pyplot(ggplot.draw(plot))
					col1.pyplot(ggplot.draw(dash.mcq_plot(data_bkdn[data_bkdn["bkdn_var"]==i],height=2)))
				elif k%3==2:
					col2.header(i)
					#col2.pyplot(ggplot.draw(plot))
					col2.pyplot(ggplot.draw(dash.mcq_plot(data_bkdn[data_bkdn["bkdn_var"]==i],height=2)))
				else:
					col3.header(i)
					#col2.pyplot(ggplot.draw(plot))
					col3.pyplot(ggplot.draw(dash.mcq_plot(data_bkdn[data_bkdn["bkdn_var"]==i],height=2)))

	elif topic=="Cross_data":
		
		#data["cash"]=data["interview_type"].apply(lambda x: 1 if x=="Cash_recipients" else 0)
		#data["IDP_in_2022"]=data["displaced_since"].apply(lambda x: np.nan if x=="Not_displaced" else 1 if x=="2022" else 0)
		#st.write([i for i in data.columns if "small_vs_big" in i])
		#data["bank"]=data
		# Find a way to allow MCQ responses probably change "/" into "_"

		data["gender_coded"]=data["gender"].apply(lambda x: 1 if x=="Male" else 0)

		variable=st.selectbox("Select the varaiable to analyse",binarys+["LIKERTS"]+likerts+mcqs_choices+
						["gender_coded"
	   ])
		regression_vars=st.multiselect("Select demographic variables for regressions",demographics)

		st.write("coucou")

		to_remove=dash.to_remove(variable,xls,mcqs_choices,mcqs,scqs,binarys,demographics,likerts)
		col1,col2=st.columns([4,1])
		to_remove2=col1.multiselect("Select other variable you would like to remove form the ML model (parent or children variables already removed)",
							[i for i in demographics+likerts+binarys+scqs+mcqs if i not in to_remove])
		to_remove += to_remove2
		tops=col2.number_input("Select number of variables to extract from ML model",3,20,6)

		if st.button("Display analysis"):
			

			st.subheader("Correlation matrix of demographics")
			fig, ax = plt.subplots()
			fig=dash.corr_matrix(data,regression_vars)
			st.pyplot()
			

			df=pd.get_dummies(data[regression_vars+[variable,"pweights"]], columns=regression_vars, drop_first=True)
			st.subheader("Regressions results")
			rsquare,summ_df, df_pred = dash.regression(df,variable)

			col1,col2=st.columns([1,1])
			col2.dataframe(summ_df)			
			try:
				col1.write("R2 = "+str(round(float(rsquare),2)))
				
			except:
				col1.write("R2 = Not robust")
			col1.scatter_chart(df_pred,y="Real",x="Predicted")


			st.markdown("---")
			st.subheader("Machine Learning Results (model trained on the all dataset)")
			
			if variable in demographics+scqs+likerts+binarys+mcqs_choices+["pweights"]: 
				df=data[demographics+scqs+binarys+likerts+mcqs_choices+["pweights"]].dropna(subset=variable).copy()
			else:
				df=data[demographics+scqs+binarys+likerts+mcqs_choices+["pweights"]+[variable]].dropna(subset=variable).copy()
			df=pd.get_dummies(df, columns=scqs+demographics, drop_first=True)
			#### to remove also line 249 remove the i for i...###
			st.write(df)
			st.write([i for i in df.columns if "interview_type" in i])
			df.drop([i for i in df.columns if "interview_type" in i],axis=1,inplace=True)


			df.drop([i for i in df if df[i].fillna(0).sum()<40 and i!="pweights"],axis=1,inplace=True)

			for i in df:
				df[i]=df[i].apply(lambda x: 99 if (x!=x or x>10) else x)
					
			for i in df:
				if 99 in df[i].unique():
					mean=df[df[i]!=99][i].mean()
					df[i]=df[i].apply(lambda x:x if x!=99 else mean)

			st.write(df)

			svalues,X,predict_df, top_features = dash.lgbm(df,variable,to_remove,[i for i in scqs if "interview_type" not in i]+mcqs,top=tops)
			col1,col2=st.columns([1,1])

			shap.summary_plot(svalues, X, feature_names=X.columns, max_display=15) # Look how to display this plot on streamlit
			col1.pyplot()

			
			col2.write("R2 = "+str(round(r2_score(predict_df["Real"], predict_df["Predicted"]),2)))
			col2.scatter_chart(predict_df,y="Real",x="Predicted")

			st.markdown("---")
			st.subheader("Comparaison regressions with or without the top 6 features from the ML model")

			df=pd.get_dummies(data[regression_vars+[variable,"pweights"]], columns=regression_vars, drop_first=True)	
			for i in top_features:
				df[i]=X[i]
			df.columns=["_".join(i.split("/")) for i in df.columns]

			rsquare_ml,summ_df_ml, df_pred_ml = dash.regression(df,variable)

			col1,col2=st.columns([1,1])

			col1.subheader("With")
			col1.write("R2 = "+str(round(float(rsquare_ml),2)))
			col1.scatter_chart(df_pred_ml,y="Real",x="Predicted")
			col1.write(summ_df_ml)

			col2.subheader("Without")
			col2.write("R2 = "+str(round(float(rsquare),2)))
			col2.scatter_chart(df_pred,y="Real",x="Predicted")
			col2.write(summ_df)


#			st.markdown("---")
#			st.subheader("Machine Learning Results (with testing and training set split 20-80)")
#			
#			df=data[demographics+scqs+likerts+binarys+["pweights"]].dropna(subset=variable).copy()
#			df=pd.get_dummies(df, columns=scqs+demographics, drop_first=True)
#			df.drop([i for i in df if df[i].sum()<40 and i!="pweights"],axis=1,inplace=True)
#
#			for i in df:
#				df[i]=df[i].apply(lambda x: 99 if (x!=x or x>10) else x)
#					
#			for i in df:
#				if 99 in df[i].unique():
#					mean=df[df[i]!=99][i].mean()
#					df[i]=df[i].apply(lambda x:x if x!=99 else mean)
#
#			svalues,X,predict_df, top_features = dash.lgbm(df,variable,to_remove,scqs+mcqs,top=tops,training=True)
#			col1,col2=st.columns([1,1])
#
#			shap.summary_plot(svalues, X, feature_names=X.columns, max_display=15) # Look how to display this plot on streamlit
#			col1.pyplot()
#			
#			col2.subheader("Prediction from the testing set (one fifth of the dataset)")
#			col2.write("R2 = "+str(round(r2_score(predict_df["Real"], predict_df["Predicted"]),2)))
#			col2.scatter_chart(predict_df,y="Real",x="Predicted")
#
			st.markdown("---")
			st.subheader("Comparaison regressions with or without the top 6 features from the ML model")

			df=pd.get_dummies(data[regression_vars+[variable,"pweights"]], columns=regression_vars, drop_first=True)	
			for i in top_features:
				df[i]=X[i]
			df.columns=["_".join(i.split("/")) for i in df.columns]

			rsquare_ml,summ_df_ml, df_pred_ml = dash.regression(df,variable)

			col1,col2=st.columns([1,1])

			col1.subheader("With")
			col1.write("R2 = "+str(round(float(rsquare_ml),2)))
			col1.scatter_chart(df_pred_ml,y="Real",x="Predicted")
			col1.write(summ_df_ml)

			col2.subheader("Without")
			col2.write("R2 = "+str(round(float(rsquare),2)))
			col2.scatter_chart(df_pred,y="Real",x="Predicted")
			col2.write(summ_df)



if __name__ == '__main__':
	main()
