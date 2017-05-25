
module KinectSkeletonV1

export
	HIP_CENTER, SPINE, SHOULDER_CENTER, HEAD, SHOULDER_LEFT,
	ELBOW_LEFT, WRIST_LEFT, HAND_LEFT, SHOULDER_RIGHT, ELBOW_RIGHT,
	WRIST_RIGHT, HAND_RIGHT, HIP_LEFT, KNEE_LEFT, ANKLE_LEFT, FOOT_LEFT,
	HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT, FOOT_RIGHT, SKELETON_POSITION_COUNT,
	nui_skeleton_names, nui_skeleton_conn,
	filter_cont, eighty2sixty

const HIP_CENTER = 1
const SPINE = 2
const SHOULDER_CENTER = 3
const HEAD = 4
const SHOULDER_LEFT = 5
const ELBOW_LEFT = 6
const WRIST_LEFT = 7
const HAND_LEFT = 8
const SHOULDER_RIGHT = 9
const ELBOW_RIGHT = 10
const WRIST_RIGHT = 11
const HAND_RIGHT = 12
const HIP_LEFT = 13
const KNEE_LEFT = 14
const ANKLE_LEFT = 15
const FOOT_LEFT = 16
const HIP_RIGHT = 17
const KNEE_RIGHT = 18
const ANKLE_RIGHT = 19
const FOOT_RIGHT = 20
const SKELETON_POSITION_COUNT = 20

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
function filter_cont(X)
	i1 = -1
	T = size(X,1)
	for i = 1:T
		if norm(X[i,:]) > 0.0
			i1 = i
			break
		end
	end
	if i1 < 0
		return X[1:-1,:]
	end
	i2 = i1
	for i = i1:T
		if norm(X[i,:]) <= 0.0
			break
		end
		i2 = i
	end
	X[i1:i2,:]
end

function eighty2sixty(X80)
	@assert size(X80,2) == 80
	T = size(X80,1)

	X60 = zeros(T,60)
	for i = 1:T
		for j=0:19
			for d=1:3
				X60[i, j*3+d] = X80[i, j*4+d]
			end
		end
	end
	X60
end

end

