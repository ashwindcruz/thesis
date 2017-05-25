
module KinectVisualizeV1

using PyCall
using PyPlot
@pyimport mpl_toolkits.mplot3d.art3d as art3d

using Reel
using ProgressMeter

using KinectSkeletonV1

export
	visualize_skeleton,
	visualize_skelseq

# xf: vector with 60 or 80 dimensions containing the joint xyz locations
function visualize_skeleton(xf; title="", fsize=6)
	if length(xf) == 80
		X = reshape(xf, 4, SKELETON_POSITION_COUNT)
	elseif length(xf) == 60
		X = reshape(xf, 3, SKELETON_POSITION_COUNT)
	elseif mod(length(xf),3) == 0	# subset of bones
		X = reshape(xf, 3, int(length(xf)/3))
	else
		error("Invalid number of dimensions in skeleton!")
	end

	fig = plt[:figure](figsize=(fsize,fsize))
	axes = fig[:add_subplot](111, projection="3d")

	axes[:set_xlabel]("\$x\$ (m)")
	axes[:set_ylabel]("\$z\$ (m)")
	axes[:set_zlabel]("\$y\$ (m)")

	axes[:set_xlim]([-1.5, 1.5])
	axes[:set_zlim]([-1.0, 1.5])
	axes[:set_ylim]([2.0, 3.5])

	axes[:set_title](title)

	# Visualize end points
	axes[:scatter](X[1,:], X[3,:], X[2,:], c="k", marker="o")

	function is_valid_bone(X, bi)
		b1 = nui_skeleton_conn[bi][1]
		b2 = nui_skeleton_conn[bi][2]
		b1 <= size(X,2) && b2 <= size(X,2)
	end
	bonecount = sum(map(bi -> is_valid_bone(X, bi), 1:length(nui_skeleton_conn)))

	# Visualize skeleton
	S = zeros(bonecount,2,3)
	si = 1
	for bi = 1:length(nui_skeleton_conn)
		if is_valid_bone(X, bi)
			S[si,1,1] = X[1, nui_skeleton_conn[bi][1]]
			S[si,1,2] = X[3, nui_skeleton_conn[bi][1]]
			S[si,1,3] = X[2, nui_skeleton_conn[bi][1]]
			S[si,2,1] = X[1, nui_skeleton_conn[bi][2]]
			S[si,2,2] = X[3, nui_skeleton_conn[bi][2]]
			S[si,2,3] = X[2, nui_skeleton_conn[bi][2]]
			si += 1
		end
	end
	lc3 = art3d.Line3DCollection(S, colors=(0.0,0.0,0.0,1.0))
	axes[:add_collection](lc3)
	axes[:view_init](elev=20., azim=50)

	fig[:tight_layout]()
	fig, axes
end

function visualize_skelseq(X; fps=30)
	T = size(X,1)
	@assert T >= 1

	#Reel.set_output_type("mp4")

	plt.ioff()
	frames = Frames(MIME("image/png"), fps=fps)
	p = Progress(T, 1, "Rendering...", 45)
	for ti=1:T
		fig, axes = visualize_skeleton(X[ti,:]; title="$ti of $T")
		push!(frames, fig)
		next!(p)
	end
	plt.ion()
	frames
end

end

