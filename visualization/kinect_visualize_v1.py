import pdb

import mpl_toolkits.mplot3d.art3d as art3d
from kinect_skeleton_v1 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_skeleton(xf, title="", fsize=6):
	if len(xf) == 80:
		X = np.reshape(xf, (4, SKELETON_POSITION_COUNT+1))
	elif len(xf) == 60:
		X = np.reshape(xf, (3, SKELETON_POSITION_COUNT+1))
	elif (len(xf)%3) == 0:	# subset of bones
		X = np.reshape(xf, (3, int(length(xf)/3)))
	else:
		print('Invalid number of dimensions in skeleton')
		print(len(xf))
		return -1
		#error("Invalid number of dimensions in skeleton!")
	#pdb.set_trace()
	fig = plt.figure(figsize=(fsize,fsize))
	axes = fig.add_subplot(111, projection="3d")

	axes.set_xlabel("\$x\$ (m)")
	axes.set_ylabel("\$z\$ (m)")
	axes.set_zlabel("\$y\$ (m)")

	# axes.set_xlim([-1.5, 1.5])
	# axes.set_zlim([-1.0, 1.5])
	# axes.set_ylim([2.0, 3.5])

	axes.set_xlim([-3, 3])
	axes.set_zlim([-2, 3])
	axes.set_ylim([4, 7])


	axes.set_title(title)

	# Visualize end points
	axes.scatter(X[0,:], X[2,:], X[1,:], c="k", marker="o")
	
	#bonecount = sum(map(bi -> is_valid_bone(X, bi), 1:length(nui_skeleton_conn)))
	bonecount = sum([is_valid_bone(X,bi) for bi in range(len(nui_skeleton_conn))])
	#print(bonecount)
	# Visualize skeleton
	S = np.zeros((bonecount,2,3))
	#pdb.set_trace()
	si = 0
	for bi in range(len(nui_skeleton_conn)): #= 1:length(nui_skeleton_conn)
		if is_valid_bone(X, bi):
			#print('si is %d and bi is %d' %(si, bi))
			# S[si,0,0] = X[0, nui_skeleton_conn[bi][0]]
			# S[si,0,1] = X[2, nui_skeleton_conn[bi][0]]
			# S[si,0,2] = X[1, nui_skeleton_conn[bi][0]]
			# S[si,1,0] = X[0, nui_skeleton_conn[bi][1]]
			# S[si,1,1] = X[2, nui_skeleton_conn[bi][1]]
			# S[si,1,2] = X[1, nui_skeleton_conn[bi][1]]

			S[si,0,0] = X[0, nui_skeleton_conn[bi][0]]
			S[si,0,1] = X[2, nui_skeleton_conn[bi][0]]
			S[si,0,2] = X[1, nui_skeleton_conn[bi][0]]
			#pdb.set_trace()
			S[si,1,0] = X[0, nui_skeleton_conn[bi][1]]
			S[si,1,1] = X[2, nui_skeleton_conn[bi][1]]
			S[si,1,2] = X[1, nui_skeleton_conn[bi][1]]
			

			si += 1
	#print('out')	
	
	lc3 = art3d.Line3DCollection(S, colors=(0.0,0.0,0.0,1.0))
	axes.add_collection(lc3)
	axes.view_init(elev=20., azim=50)

	fig.tight_layout()
	return fig, axes

def is_valid_bone(X, bi):
	b1 = nui_skeleton_conn[bi][0]
	b2 = nui_skeleton_conn[bi][1]
		#b1 <= size(X,2) && b2 <= size(X,2)
	return ( (b1<=X.shape[1]) and (b2<=X.shape[1]) )
	