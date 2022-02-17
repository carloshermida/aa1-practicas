
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.2
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


using Statistics
using DelimitedFiles
using Flux

############################## FUNCIONES ##############################

##### oneHotEncoding

# Función principal
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes)
    if length(classes) == 2
        feature=feature.==classes[1]
        reshape(feature, (length(feature),1))
        return feature;
    else length(classes) > 2
        bool_matrix=falses(length(feature),length(classes))
        for i=1:length(classes)
            bool_matrix[:,i]=(feature.==classes[i])
        end
    return bool_matrix;
    end
end

# Función sobrecargada 1
oneHotEncoding(feature::AbstractArray{<:Any,1})=(classes=unique(feature);oneHotEncoding(feature, classes))

# Función sobrecargada 2
function oneHotEncoding(feature::AbstractArray{Bool,1})
    classes=unique(feature)
    if length(classes) == 2
        feature=feature.==classes[1]
        reshape(feature, (length(feature),1))
        return feature;
    end
end


##### Máximos y mínimos

# Función auxiliar
function calculateMinMaxNormalizationParameters(numerics::AbstractArray{<:Real,2})
    m= maximum(numerics, dims=1)
    mi=minimum(numerics, dims=1)
    return (m, mi)
end

# Función sobrecargada 1
function normalizeMinMax!(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    return (numerics.-MinMax[2])./(MinMax[1].-MinMax[2])
end

# Función sobrecargada 2
normalizeMinMax!(numerics::AbstractArray{<:Real,2})=(normalizeMinMax!(numerics, calculateMinMaxNormalizationParameters(numerics)))

# Función sobrecargada 3
function normalizeMinMax(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    return (copia.-MinMax[2])./(MinMax[1].-MinMax[2])
end

# Función sobrecargada 4
normalizeMinMax(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeMinMax(copia, calculateMinMaxNormalizationParameters(numerics)))


##### Media y desviación típica

# Función principal
function calculateZeroMeanNormalizationParameters(numerics::AbstractArray{<:Real,2})
    media=mean(numerics, dims=1)
    sd= std(numerics, dims=1)
    return (media, sd)
end

# Función sobrecargada 1
function normalizeZeroMean!(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    return (numerics.-media_sd[1])./media_sd[2]
end

# Función sobrecargada 2
normalizeZeroMean!(numerics::AbstractArray{<:Real,2})=(normalizeZeroMean!(numerics,calculateZeroMeanNormalizationParameters(numerics_haber)))

# Función sobrecargada 3
function normalizeZeroMean(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    return (copia.-media_sd[1])./media_sd[2]
end

# Función sobrecargada 4
normalizeZeroMean(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeZeroMean(copia,calculateZeroMeanNormalizationParameters(numerics)))


##### classifyOutputs

# Función principal
function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold = 0.5)
    columns = size(outputs)[2]
    if columns == 1
        outputs = outputs .>= threshold
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] = outputs[indicesMaxEachInstance] .= true;
    end
    return outputs
end


##### accuracy

# Función sobrecargada 1
function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1})
    classComparison = targets .== outputs
    return sum(classComparison)/length(classComparison)
end

# Función sobrecargada 2
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    rows = size(outputs)[1]
    columns = size(outputs)[2]
    if columns == 1
        targets = (targets[:,1])
        outputs = (outputs[:,1])
        return accuracy(targets, outputs)
    else
        classComparison = falses(rows,1)
        for row = 1:rows
            classComparison[row,1] = targets[row,:] == outputs[row,:]
        end
        return sum(classComparison)/length(classComparison)
    end
end

# Función sobrecargada 3
function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, threshold = 0.5)
    outputs = outputs .>= threshold
    return accuracy(targets, outputs)
end

# Función sobrecargada 4
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    rows = size(outputs)[1]
    columns = size(outputs)[2]
    if columns == 1
        targets = (targets[:,1])
        outputs = (outputs[:,1])
        return accuracy(targets, outputs)
    else
        outputs = classifyOutputs(outputs)
        print(outputs)
        return accuracy(targets, outputs)
    end
end


### red neuronal artificial

function rna(topology::AbstractArray{<:Int,1}, n_input, n_output)
    ann = Chain();
    numInputsLayer = n_input
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end
    if n_output <= 2
        ann = Chain(ann...,  Dense(numInputsLayer, 1, σ) );
    else
        ann = Chain(ann...,  Dense(numInputsLayer, ###n_output, identity), softmax);
    end
    return ann
end

dataset_haber = readdlm("haberman.data",',');
dataset_haber = dataset_haber'
feature_haber = dataset_haber[4,:]
feature_haber = convert(AbstractArray{<:Any,1},feature_haber);
target = oneHotEncoding(feature_haber)
numerics_haber = dataset_haber[1:3,:]'
input = normalizeMinMax!(numerics_haber)
x = input'

red = rna(([3]), 3, ###2)
red(x)


##
############################### CÓDIGO ###############################

##### oneHotEncoding

### IRIS
dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
classes_iris = unique(dataset_iris[5,:])
feature_iris = dataset_iris[5,:]
feature_iris = convert(AbstractArray{<:Any,1}, feature_iris)

feature_iris_1 = oneHotEncoding(feature_iris, classes_iris)    # Función principal
feature_iris_2 = oneHotEncoding(feature_iris)                  # Función sobrecargada 1

### HABERMAN
dataset_haber = readdlm("haberman.data",',');
dataset_haber = dataset_haber'
classes_haber = unique(dataset_haber[4,:])
feature_haber = dataset_haber[4,:]
feature_haber = convert(AbstractArray{<:Any,1},feature_haber);

feature_haber_1 = oneHotEncoding(feature_haber, classes_haber)  # Función principal
feature_haber_2 = oneHotEncoding(feature_haber)                 # Función sobrecargada 1
feature_haber_3 = oneHotEncoding(feature_haber_2)               # Función sobrecargada 2

##### Máximos y mínimos
numerics_haber=dataset_haber[1:3,:]'
normalizeMinMax!(numerics_haber, calculateMinMaxNormalizationParameters(numerics_haber))    # Función sobrecargada 1
normalizeMinMax!(numerics_haber)                                                            # Función sobrecargada 2
normalizeMinMax(numerics_haber, calculateMinMaxNormalizationParameters(numerics_haber))     # Función sobrecargada 3
normalizeMinMax(numerics_haber)                                                             # Función sobrecargada 4

##### Media y desviación típica
normalizeZeroMean!(numerics_haber,calculateZeroMeanNormalizationParameters(numerics_haber)) # Función sobrecargada 1
normalizeZeroMean!(numerics_haber)                                                          # Función sobrecargada 2
normalizeZeroMean(numerics_haber,calculateZeroMeanNormalizationParameters(numerics_haber))  # Función sobrecargada 3
normalizeZeroMean(numerics_haber)                                                           # Función sobrecargada 4

##### Normalizar por columnas
d = dataset_haber[:,1:3]
m = maximum(d, dims=1)
mi = minimum(d, dims=1)
med = mean(d, dims=1)
sd = std(d, dims=1)

### Forma 1:
norm = (d.-med)./sd
for i=1:size(d,2)
    if sd[i]==0
        norm[:,i].=0
    end
end

### Forma 2:
norm_2 = (d.-mi)./(m.-mi)
for i=1:size(d,2)
    if mi[i]==m[i]
        norm_2[:,i].=0
    end
end

##### Test accuracy functions

targets = convert(AbstractArray{Bool,2}, reshape([1,0,0,1], (2,2)))
outputs = reshape([0,2,0,1], (2,2))
accuracy(targets, outputs)
