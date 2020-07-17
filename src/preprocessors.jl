
identity() = x -> x

###
# rescaling each observation to fill [0,1]
function scaling()
	return function(X)
		for x in eachslice(X,dims=ndims(X))
			m,M = extrema(x)
			x .= @. (x - m) / (M - m)
		end
	end
end

function centering()
	return function(X)
		for x in eachslice(X,dims=ndims(X))
			μ = mean(x)
			x .= @. x - μ
		end
	end
end

function standardizing()
	return function(X)
		for x in eachslice(X,dims=ndims(X))
			μ,σ = mean(x),std(x)
			x .= @. (x-μ)/σ
		end
	end
end

function whitening(X)
	numobs = size(X,ndims(X))
	Y = reshape(X,:,numobs)
	U,σ,Vt = svd(Y)
	Ut = transpose(U)
	return function(X)
		Y = Ut*reshape(X,:,size(X,ndims(X)))
		Y ./= σ
		X[:] .= Y[:]
	end
end	
# ###
# # standardize images
# x̄ = mean(train_x,dims=4)
# train_x .-= x̄
# test_x .-= x̄
# σ = std(train_x,dims=4)
# train_x ./= σ
# test_x ./= σ

