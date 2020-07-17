
identity!(x) = x

###
# rescaling each observation to fill [0,1]
function scaler!(X)
	for x in eachslice(X,dims=ndims(X))
		m,M = extrema(x)
		x .= @. (x - m) / (M - m)
	end
end

function center!(X)
	for x in eachslice(X,dims=ndims(X))
		μ = mean(x)
		x .= @. x - μ
	end
end

function standardize!(X)
	for x in eachslice(X,dims=ndims(X))
		μ,σ = mean(x),std(x)
		x .= @. (x-μ)/σ
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

