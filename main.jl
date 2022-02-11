
function cat_to_num(dataset, cat_indexes)
    dataset = convert(Array{Any},dataset);
    for cat_index in cat_indexes
        categorical = dataset[:,cat_index];
        cat_options = unique(categorical, dims = 1);

        if length(cat_options) == 2
            let normal = "0";
                for i = 1:2
                    dataset[:,cat_index] = replace(dataset[:,cat_index],
                     cat_options[i,1] => normal);
                    normal = "1";
                end
            end
        else
            let normal = string.(repeat("0",length(cat_options)-1), "1");
                for i = 1:length(cat_options)
                    dataset[:,cat_index] = replace(dataset[:,cat_index],
                     cat_options[i,1] => normal);
                    normal = string.(normal[2:end], "0");
                end
            end
        end
    end
    return dataset;
end

########## MODIFICAR ESTA FUNCION COMO LA PIDE EL ENUNCIADO
########## Y QUE DEVUELVA UNA MATRIZ DE SALIDA

#######################################
### INDICA EL DATASET Y LOS INDICES ###
#######################################
# Pasar a binario las categóricas
using DelimitedFiles

dataset_iris = readdlm("iris.data",',');
cat_indexes_iris = 5;
dataset_iris = cat_to_num(dataset_iris, cat_indexes_iris);

dataset_haber = readdlm("haberman.data",',');
cat_indexes_haber = 4;
dataset_haber = cat_to_num(dataset_haber, cat_indexes_haber);
#######################################
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
