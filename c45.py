import numpy as np

def c45ContinousHandling(df, column_name, entropy):
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

            gains[i] = entropy - subset1_prob*CountEntropy(subset1) - subset2_prob*CountEntropy(subset2)
        
        threshold = values[gains.index(max(gains))]
        df[column_name] = np.where(df[column_name] <= threshold, "<="+str(threshold), ">"+str(threshold))

        return df

def CountEntropy(subset) :
    return 0
