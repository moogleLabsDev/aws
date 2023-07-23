import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd

def convert (d_lim):
	def support (lim_d):
        
		df2=pd.read_csv('min_nearby_distances.csv')
		df1=pd.read_csv('sample_timestream.csv')
		df2 = df2[df2['Device_id3'].notna()]
		df2 = df2[df2['Device_id2'].notna()]
		return df1,df2

	df1,df2=support(d_lim)
    
	def model1 ():
	
		map_location = "cuda:@" if torch.cuda.is_available() else "cpu"
		device = torch.device(map_location)

		MODEL_PATH = 'all_4_inputs_4_nodes_50000.pth'

		num_nodes = 4
		n_input_per_node = 4
		n_input = num_nodes * n_input_per_node
		n_hidden = 30
		n_out = 1

		model = nn.Sequential(nn.Linear(n_input, n_hidden),
		nn.ReLU(),
		nn.Linear(n_hidden, n_out),
		nn.Sigmoid() )

		model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(map_location) ))

		model.eval()
		model.to(device)

		return model
    
	model=model1()
    
	df=df1.copy()
	mes=['discrepancy_gas_new','discrepancy_particle','ic_score','PM4P0'] # made a list of Device name for save the value based on this order
	aaa1,aaa2=[],[]
	val1=df.stationID.drop_duplicates()	# from the time stream table we remove the duplicates value to extrace the lates saved device ids shorted by last 15 mins
	for k in val1:
		for j,h in enumerate(df2.Device_id):
			if h == k: # check short distance file device id with respect to time stream data device id
				aaa1.append([df2[j:j+1]['Device_id'].item(),df2[j:j+1]['Device_id1'].item(),df2[j:j+1]['Device_id2'].item(),df2[j:j+1]['Device_id3'].item()])
		# itterate through short distance file to extract 3 shortest distance device ids and add it in list 
		# this will contain 2 set of device id which matches data
		
	for d in aaa1:
		aaa2.append(d[0])

	aa1,aa2,aa3=[],[],[]

	for sd in aaa1:
		for ds in sd:
			for bb in mes:
				aa1.append(np.where((df1.stationID == ds) & (df1.measure_name == bb)))
		# this list contain index value of device ids of all near distance also.
		 
		if sum([1 for x in aa1 if len(x[0])==0]) == 0:
			aa2.append(aa1)
			aa3.append(ds)
		aa1=[]
    
	sam1,sam2,sam3=[],[],[]
    
	for e in aa2:
		dd=e[0][0][0]
		
		for f,s in enumerate(e):
			ss=s[0][0]
			sam1.append(float(df1[ss:ss+1]['measure_value::double'].item()))	# this will extract the measure values of the respective device ids
		sam3.append(df1[dd:dd+1]['time'].item())				# this list will get time stamp of the specific device id from time stream table
		sam2.append(sam1)
		sam1=[]

	out=[]
	for x in range(len(sam2)):
		samp=torch.tensor(sam2[x])
		out.append(model(samp))

	out1=[]
	for i in out:
		i=i.tolist()
		out1.append(i)

	out=[j for sub in out1 for j in sub]

	dataout=pd.DataFrame({'Device_id':aa3,'PRIMARY_KEY':sam3,'p_fire':out})
	dataout.to_csv("result.csv")
	#print(dataout)
	
	return dataout

res=convert(1000)    
print("file save as result.csv in present working folder succesfully !!!")
    
    
