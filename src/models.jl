function baremodel()

	return Chain(
   
		# input 32x32x3
		Conv((5,5), 3=>32, pad=(2,2), relu),  # now 32x32x32
		MaxPool((3,3),pad=(1,1),stride=2),  # now 16x16x32

		Conv((5,5), 32=>32, pad=(2,2), relu),
		MeanPool((3,3),pad=(1,1),stride=2),  # now 8x8x32

		Conv((5,5), 32=>64, pad=(2,2), relu),
		MeanPool((3,3),pad=(1,1),stride=2),  # now 4x4x64

		Conv((4,4), 64=>64, pad=(0,0), relu),  # now 1x1x64 

		# Reshape tensor into 2d
		x -> reshape(x, :, size(x, 4)),
		Dense(64, 10),

		# Finally, softmax to get probabilities
		softmax,
	)

end

function vggmodel()

	return Chain(
   
		# input 32x32x3
		Conv((3,3), 3=>32, pad=(1,1), relu),  # now 32x32x32
		Conv((3,3), 32=>32, pad=(1,1), relu),  # now 32x32x32
		MaxPool((2,2),pad=(0,0),stride=2),  # now 16x16x32

		Conv((3,3), 32=>64, pad=(1,1), relu),
		Conv((3,3), 64=>64, pad=(1,1), relu),
		MeanPool((2,2),pad=(0,0),stride=2),  # now 8x8x64

		Conv((3,3), 64=>128, pad=(1,1), relu),
		Conv((3,3), 128=>128, pad=(1,1), relu),
		MeanPool((2,2),pad=(0,0),stride=2),  # now 4x4x128

		Conv((4,4), 128=>128, pad=(0,0), relu),  # now 1x1x128 

		# Reshape tensor into 2d
		x -> reshape(x, :, size(x, 4)),
		Dense(128, 10),

		# Finally, softmax to get probabilities
		softmax,
	)

end
function dropmodel()

	return Chain(
   
		# input 32x32x3
		Conv((5,5), 3=>32, pad=(2,2), relu),  # now 32x32x32
		MaxPool((2,2),pad=(0,0),stride=2),  # now 16x16x32
		Dropout(0.15),

		Conv((5,5), 32=>32, pad=(2,2), relu),
		MeanPool((2,2),pad=(0,0),stride=2),  # now 8x8x32
		Dropout(0.15),

		Conv((5,5), 32=>64, pad=(2,2), relu),
		MeanPool((2,2),pad=(0,0),stride=2),  # now 4x4x64
		Dropout(0.15),

		Conv((4,4), 64=>64, pad=(0,0), relu),  # now 1x1x64 
		Dropout(0.35),

		# Reshape tensor into 2d
		x -> reshape(x, :, size(x, 4)),
		Dense(64, 10),

		# Finally, softmax to get probabilities
		softmax,
	)

end
