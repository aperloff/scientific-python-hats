import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from prepData import *

class random_forest_regression:

	def __init__(self,factors=[''],regressor='jet_jes_ak7',data_frame=None):
		self.regressor = regressor
		self.max_depth_low = 5
		self.max_depth_high = 100
		self.max_depth = 30
		self.n_trees = 50
		self.model = None
		self.data_frame=data_frame
		self.factors = factors
		self.score_ = None
		self.SSE = 0.

	def aggregate_score(self,data_frame):
		n=float(len(data_frame[self.regressor]))
		diff = data_frame['prediction']-data_frame[self.regressor]
		RMSE_vec = map(lambda x : x*x,diff)
		self.SSE += sum(RMSE_vec)

	def score(self,data_frame,verbose=False,append=""):
		n=float(len(data_frame[self.regressor]))
		diff = data_frame['prediction'+append]-data_frame[self.regressor]
		RMSE_vec = map(lambda x : x*x,diff)
		self.score_ = sqrt(sum(RMSE_vec)/n)
		if verbose : 
			print "RMSE:",self.score_
		return self.score_

	def fit(self,data_frame,verbose=False,append=""):
		self.model = RandomForestRegressor(max_depth=self.max_depth, random_state=2,n_estimators=self.n_trees)
		self.model.fit(data_frame[self.factors], data_frame[self.regressor])
		
		data_frame['prediction'+append] = self.model.predict(data_frame[self.factors])
		if verbose : 
			print "train:"
			self.score(data_frame,verbose,append)

	def test(self,data_frame,verbose=False,append=""):
		data_frame['prediction'+append] = self.model.predict(data_frame[self.factors])
		data_frame['residual'+append] = data_frame['prediction'+append]-data_frame[self.regressor]
		if verbose :
			print "test:"
		self.score(data_frame,verbose,append)
		
def example():
	df_perjet = jet_level_data()
	df_train,df_test = np.array_split(df_perjet,2)

	factorNames = ['jet_pt_ak7','jet_eta_ak7',
		       'jet_photonFrac_ak7','jet_electronFrac_ak7','jet_muonFrac_ak7','jet_neuHadronFrac_ak7','jet_charHadronFrac_ak7',
		       'jet_electronMult_ak7','jet_muonMult_ak7','jet_photonMult_ak7','jet_neuHadronMult_ak7','jet_charHadronMult_ak7']

	rfr = random_forest_regression(df_train,factorNames,'jet_jes_ak7')
	rfr.fit(df_train,verbose=True)
	rfr.test(df_test,verbose=True)
