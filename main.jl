
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.2
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida

############################## FUNCIONES ##############################

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes)
    """Función para pasar de variable categorica a numérica binaria"""
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

############################### CÓDIGO ###############################

##### Pasar a binario las categóricas

using DelimitedFiles
dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
classes_iris = unique(dataset_iris[5,:])
feature_iris = dataset_iris[5,:]
feature_iris = convert(Array{Any},feature_iris);
feature_iris2 = convert(AbstractArray{<:Any,1}, feature_iris)
d = oneHotEncoding(feature_iris2, classes_iris)

dataset_haber = readdlm("haberman.data",',');
dataset_haber = dataset_haber'
classes_haber = unique(dataset_haber[4,:])
feature_haber = dataset_haber[4,:]
feature_haber1 = convert(AbstractArray{<:Any,1},feature_haber);
d = oneHotEncoding(feature_haber1, classes_haber)
m = oneHotEncoding(d)
numerics_haber=dataset_haber[1:3,:]'

##### Normalización

# obtener max y min
function calculateMinMaxNormalizationParameters(numerics::AbstractArray{<:Real,2})
    m= maximum(numerics, dims=1)
    mi=minimum(numerics, dims=1)
    return (m, mi)
end
calculateMinMaxNormalizationParameters(numerics_haber)


# obtener media sd
using Statistics
function calculateZeroMeanNormalizationParameters(numerics::AbstractArray{<:Real,2})
    media=mean(numerics, dims=1)
    sd= std(numerics, dims=1)
    return (media, sd)
end

calculateZeroMeanNormalizationParameters(numerics_haber)

# normalizaciion max min 1
function normalizeMinMax!(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    return (numerics.-MinMax[2])./(MinMax[1].-MinMax[2])
end

# normalizacion max min 2
normalizeMinMax!(numerics::AbstractArray{<:Real,2})=(normalizeMinMax!(numerics, calculateMinMaxNormalizationParameters(numerics)))
normalizeMinMax!(numerics_haber)

# normalizacion max min 3
function normalizeMinMax(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    return (copia.-MinMax[2])./(MinMax[1].-MinMax[2])
end

# normalizacion max min 4
normalizeMinMax(numerics_haber, calculateMinMaxNormalizationParameters(numerics_haber))

normalizeMinMax(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeMinMax(copia, calculateMinMaxNormalizationParameters(numerics)))
normalizeMinMax(numerics_haber)

# normalizacion media sd 1
function normalizeZeroMean!(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    return (numerics.-media_sd[1])./media_sd[2]
end

# normalizacion media sd 2
normalizeZeroMean!(numerics_haber,calculateZeroMeanNormalizationParameters(numerics_haber))

normalizeZeroMean!(numerics::AbstractArray{<:Real,2})=(normalizeZeroMean!(numerics,calculateZeroMeanNormalizationParameters(numerics_haber)))
normalizeZeroMean!(numerics_haber)

# normalizacion media sd 3
function normalizeZeroMean(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    return (copia.-media_sd[1])./media_sd[2]
end

# normalizacion media sd 4
normalizeZeroMean(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeZeroMean(copia,calculateZeroMeanNormalizationParameters(numerics)))

normalizeZeroMean(numerics_haber)



# Normalizar por columnas de la primera forma:
norm = (d.-media)./sd
for i=1:size(d,2)
    if sd[i]==0
        norm[:,i].=0
    end
end
norm

# Normalizar por columnas de la segunda forma:
norm_2 = (d.-mi)./(m.-mi)
for i=1:size(d,2)
    if mi[i]==m[i]
        norm_2[:,i].=0
    end
end
norm_2
