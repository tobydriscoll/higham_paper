
using DrWatson
@quickactivate "higham_paper"

using LinearAlgebra,Statistics
using Flux,Flux.Data, CUDA
using Flux: onehotbatch, onecold, crossentropy, throttle
using MLDatasets,MLBase

include("models.jl")
include("preprocessors.jl")

###
# Grab training and test data
train_x,train_y = CIFAR10.traindata(Float32);

## 
# hyperparameters
preprocess! = whitening(train_x[:,:,:,40001:50000])
opt = ADAM()
modelgen = dropmodel
#loss(x,y) = crossentropy(model(x),y)
function loss(x, y)
    ŷ = model(x)
    ŷ = max.(eps(eltype(ŷ)),ŷ)
    z = crossentropy(ŷ, y)
    return isnan(z) ? convert(typeof(z),16) : z
end
description = "dropmodel_standardized_ADAM_crossentropy"
hypparam = @dict description preprocess! opt modelgen loss

##
preprocess!(train_x)

train_hot = onehotbatch(train_y,0:9);
data = DataLoader((gpu(train_x),gpu(train_hot)),batchsize=100);

test_x,test_y = CIFAR10.testdata(Float32);
preprocess!(test_x)
test_x = gpu(test_x);
test_hot = gpu(onehotbatch(test_y,0:9));

###
# remainder of setup
acc(x,y) = mean( cpu(model(x)) .== cpu(y) )
model = gpu(modelgen());
progress() = loss(test_x,test_hot)
cback() = @show(progress())

###
# train
Flux.testmode!(model,false)   # enable dropout
Flux.@epochs 20 Flux.train!(loss,params(model),data,opt,cb=throttle(cback,5))
Flux.testmode!(model,true)    # disable dropout

###
# assess results
prediction = onecold(cpu(model(test_x)))
C = confusmat(10,test_y.+1,prediction)

precision = [ C[k,k]/sum(C[:,k]) for k in 1:10 ]
recall = [ C[k,k]/sum(C[k,:]) for k in 1:10 ]
accuracy = sum(diag(C))/length(test_y)

result = copy(hypparam)
result[:model] = cpu(model)
result[:confusion] = C 
result[:precision] = precision
result[:recall] = recall
result[:accuracy] = accuracy

safesave(datadir("simulations", savename(hypparam, "bson")), result)