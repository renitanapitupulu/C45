
import pandas as pd
import numpy as np
import operator

eps = np.finfo(float).eps

play_tennis = pd.read_csv("play-tennis.csv")
print(play_tennis)
print()


def globalEntropy(df):
	entropy = 0
	vals = df.iloc[:,-1].unique()
	len_vals = len(df.iloc[:,-1])
	for val in vals:
		p = df.iloc[:,-1].value_counts()[val]/len_vals
		entropy = entropy + -p*safe_log2(p)
	return entropy

def safe_log2(x):
    if x <= 0:
        return 0
    return np.log2(x)

def attrEntropy(df,attrName):
	attrs = df.iloc[:,-1].unique()
	attrVals = df[attrName].unique()

	#print (attrs)
	#print(attrVals)
	
	entropy = 0
	for attrVal in attrVals:
		ent = 0
		for attr in attrs:
			sv = len(df[attrName][df.iloc[:,-1] == attr][df[attrName] == attrVal])
			s = len(df[attrName][df[attrName] == attrVal])
			frac = sv/(s)
			ent += -frac*safe_log2(frac)
		t = s/len(df)
		entropy += -t*ent
	
	return (abs(entropy))

def informationGain(rootEntropy, attrEntropy):
	return rootEntropy - attrEntropy

def bestAttr(df, cols):
	gains = {}
	for col in cols:
		gains[col] = informationGain(globalEntropy(df), attrEntropy(df, col))
	return max(gains, key=gains.get)

def filterTab(df, attr, val):
	return (df.loc[df[attr]==val])

def check_all_attr(datas, wished_value):
	for data in datas:
		if data != wished_value:
			return False
	return True

def id3(df, targetAttr, attrs, tree = None):

	if tree is None:
		tree = {}
	
	if check_all_attr(df[targetAttr], 'Yes'):
		tree = {'Yes'}
		return tree
	if check_all_attr(df[targetAttr], 'No'):
		tree = {'No'}
		return tree

	attr = bestAttr(df, attrs)
	attrs.remove(attr)
	tree[attr] = {}
	

	for attrVal in df[attr].unique():

		tree[attr][attrVal] = {}
		new_df = df.loc[df[attr] == attrVal]
		tree[attr][attrVal] = id3(new_df, targetAttr, attrs, tree[attr])
		
	return tree	

def bestAttrc45(df, cols):
	gains = {}
	nonobject = {}
	for col in cols:
		if df[col].dtype != 'object' :
			nonobject[col], gains[col] = c45ContinousHandling(df, col)
		else :
			gains[col] = informationGain(globalEntropy(df), attrEntropy(df, col))
	
	if max(gains, key=gains.get) in nonobject :
		df.loc[:,max(gains, key=gains.get)] = nonobject[max(gains, key=gains.get)]

	return max(gains, key=gains.get)

def c45ContinousHandling(df, column_name):
    values = sorted(df[column_name].unique())

    if len(values) == 1:
        threshold = values[0]
    
    else :
        gains = [0 for i in range (len(values)-1)]
        for i in range(len(values)-1) :
            threshold = values[i]
            
            subset1 = df[df[column_name] <= threshold]
            subset2 =  df[df[column_name] > threshold]

            subset1_prob = len(subset1) / len(df[column_name])
            subset2_prob = len(subset2) / len(df[column_name])
            
            gains[i] = globalEntropy(df) - subset1_prob*globalEntropy(subset1) - subset2_prob*globalEntropy(subset2)
            

    winner_gain = max(gains)
    threshold = values[gains.index(max(gains))] 
    temp = np.where(df[column_name] <= threshold, "<="+str(threshold), ">"+str(threshold))
    
    return temp, winner_gain

def c45(df, targetAttr, attrs, tree = None):

	if tree is None:
		tree = {}
	
	if check_all_attr(df[targetAttr], 'Yes'):
		tree = {'Yes'}
		return tree
	if check_all_attr(df[targetAttr], 'No'):
		tree = {'No'}
		return tree

	attr = bestAttrc45(df, attrs)
	attrs.remove(attr)
	tree[attr] = {}
	

	for attrVal in df[attr].unique():

		tree[attr][attrVal] = {}
		new_df = df.loc[df[attr] == attrVal]
		tree[attr][attrVal] = c45(new_df, targetAttr, attrs, tree[attr])
		
	return tree

#print(informationGain(globalEntropy(play_tennis), attrEntropy(play_tennis, 'outlook')))
#bestAttr(play_tennis)
#print(bestAttr(play_tennis))
#print(filterTab(play_tennis, 'outlook', 'Sunny'))
#print(check_all_attr(play_tennis.loc[play_tennis['outlook'] == 'Overcast'].iloc[:,-1], 'no'))

#print(id3(play_tennis, 'play', list(['outlook', 'temp', 'humidity', 'wind'])))
print(c45(play_tennis, 'play', list(['outlook', 'temp', 'humidity', 'wind'])))