import pdb

# HIP_CENTER = 1
# SPINE = 2
# SHOULDER_CENTER = 3
# HEAD = 4
# SHOULDER_LEFT = 5
# ELBOW_LEFT = 6
# WRIST_LEFT = 7
# HAND_LEFT = 8
# SHOULDER_RIGHT = 9
# ELBOW_RIGHT = 10
# WRIST_RIGHT = 11
# HAND_RIGHT = 12
# HIP_LEFT = 13
# KNEE_LEFT = 14
# ANKLE_LEFT = 15
# FOOT_LEFT = 16
# HIP_RIGHT = 17
# KNEE_RIGHT = 18
# ANKLE_RIGHT = 19
# FOOT_RIGHT = 20
# SKELETON_POSITION_COUNT = 20

HIP_CENTER = 0
SPINE = 1
SHOULDER_CENTER = 2
HEAD = 3
SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7
SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11
HIP_LEFT = 12
KNEE_LEFT = 13
ANKLE_LEFT = 14
FOOT_LEFT = 15
HIP_RIGHT = 16
KNEE_RIGHT = 17
ANKLE_RIGHT = 18
FOOT_RIGHT = 19
SKELETON_POSITION_COUNT = 19

nui_skeleton_names = ["HIP_CENTER", "SPINE", "SHOULDER_CENTER",
	"HEAD", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT", "HAND_LEFT",
	"SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT", "HAND_RIGHT",
	"HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
	"HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT"]

# Skeleton bone structure
nui_skeleton_conn = [
	(HIP_CENTER, SPINE),
	(SPINE, SHOULDER_CENTER),
	(SHOULDER_CENTER, HEAD),
	# Left arm
	(SHOULDER_CENTER, SHOULDER_LEFT),
	(SHOULDER_LEFT, ELBOW_LEFT),
	(ELBOW_LEFT, WRIST_LEFT),
	(WRIST_LEFT, HAND_LEFT),
	# Right arm
	(SHOULDER_CENTER, SHOULDER_RIGHT),
	(SHOULDER_RIGHT, ELBOW_RIGHT),
	(ELBOW_RIGHT, WRIST_RIGHT),
	(WRIST_RIGHT, HAND_RIGHT),
	# Left leg
	(HIP_CENTER, HIP_LEFT),
	(HIP_LEFT, KNEE_LEFT),
	(KNEE_LEFT, ANKLE_LEFT),
	(ANKLE_LEFT, FOOT_LEFT),
	# Right leg
	(HIP_CENTER, HIP_RIGHT),
	(HIP_RIGHT, KNEE_RIGHT),
	(KNEE_RIGHT, ANKLE_RIGHT),
	(ANKLE_RIGHT, FOOT_RIGHT)]

# Extract the first fully observed continuous subsequence from X
# def filter_cont(X):
# 	i1 = -1
# 	T = size(X,1)
# 	for i in range(T): #1:T
# 		if norm(X[i,:]) > 0.0
# 			i1 = i
# 			break
# 		end
# 	end
# 	if i1 < 0
# 		return X[1:-1,:]
# 	end
# 	i2 = i1
# 	for i in range(i1,T): #i1:T
# 		if norm(X[i,:]) <= 0.0
# 			break
# 		end
# 		i2 = i
# 	end
# 	return X[i1:i2,:]

# def eighty2sixty(X80):
# 	T = size(X80,1)

# 	X60 = zeros(T,60)
# 	for i in range(T): #= 1:T
# 		for j in range(20)      #=0:19
# 			for d in range(3):      #=1:3
# 				X60[i, j*3+d] = X80[i, j*4+d]
# 			end
# 		end
# 	end
# 	X60
# end