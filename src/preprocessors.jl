
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
	μ = mean(Y,dims=2)
	Y .-= μ
	#U,σ,Vt = svd(Y)
	#σ /= sqrt(numobs)
	λ,U = eigen(cov(Y'))
	W = Diagonal(1 ./sqrt.(λ))*transpose(U)

	return function(X)
		#Y = Ut*(reshape(X,:,size(X,ndims(X))) .- μ)
		#Y ./= σ
		Y = W*(reshape(X,:,size(X,ndims(X))) .- μ)
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

