
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.1
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


# Funcion para pasar de variable categorica a numérica binaria
function cat_to_num(dataset, target_index)
    dataset = convert(Array{Any},dataset);
    target = dataset[:,target_index];
    target_options = unique(target, dims = 1);

    if length(target_options) == 2
        let code = "0";
            for i = 1:2
                dataset[:,target_index] = replace(dataset[:,target_index],
                 target_options[i,1] => code);
                code = "1";
            end
        end
    else
        let code = string.(repeat("0",length(target_options)-1), "1");
            for i = 1:length(target_options)
                dataset[:,target_index] = replace(dataset[:,target_index],
                 target_options[i,1] => code);
                code = string.(code[2:end], "0");
            end
        end
    end
    return dataset;
end

# Pasar a binario las categóricas
using DelimitedFiles

dataset_iris = readdlm("iris.data",',');
target_index_iris = 5;
dataset_iris = cat_to_num(dataset_iris, target_index_iris);

dataset_haber = readdlm("haberman.data",',');
target_index_haber = 4;
dataset_haber = cat_to_num(dataset_haber, target_index_haber);

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



########avances de hoy
function oneHotEncoding(feature, classes)
    feature = convert(AbstractArray{Any},feature);
    #target_options = unique(classes, dims = 1);

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

# Pasar a binario las categóricas
using DelimitedFiles

dataset_iris = readdlm("Downloads/trabajo_ti/iris.data",',');
reshape(dataset_iris, (5,150))
permutedims(dataset_iris)
adjoint(dataset_iris,)
classes=unique(dataset_iris[5,:])


target_index_iris = 5;
dataset_iris = cat_to_num(dataset_iris, target_index_iris);

##############################################################

dataset_haber = readdlm("Downloads/trabajo_ti/haberman.data",',');
dataset_haber=dataset_haber'

classes=unique(dataset_haber[4,:])
#target_options = unique(classes, dims = 1)

feature=dataset_haber[4,:]
feature = convert(Array{Any},feature);
d=oneHotEncoding(feature, classes)
#target_options = unique(classes, dims = 1);
