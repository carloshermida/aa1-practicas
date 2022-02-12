
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.2
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida

############################## FUNCIONES ##############################

function oneHotEncoding(feature, classes)
    """Función para pasar de variable categorica a numérica binaria"""
    feature = convert(AbstractArray{Any},feature);
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
#oneHotEncoding(outputs::AbstractArray{<:Any,1}, feature::AbstractArray{Any})=(classes=unique(feature);oneHotEncoding(feature, classes))

# Función sobrecargada 2
function oneHotEncoding(feature)
    feature=convert(AbstractArray{Bool,1}, feature)
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
feature_iris = oneHotEncoding(feature_iris, classes_iris)

dataset_haber = readdlm("haberman.data",',');
dataset_haber = dataset_haber'
classes_haber = unique(dataset_haber[4,:])
feature_haber = dataset_haber[4,:]
feature_haber = convert(Array{Any},feature_haber);
feature_haber = oneHotEncoding(feature_haber, classes_haber)

##### Normalización

# Para obtener las filas con las medias, maximos, mínimos y sd
d = dataset_haber[:,1:3]
m= maximum(d, dims=1)
mi=minimum(d, dims=1)
using Statistics
media=mean(d, dims=1)
sd= std(d, dims=1)

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
