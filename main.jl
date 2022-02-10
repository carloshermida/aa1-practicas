
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

#######################################
### INDICA EL DATASET Y LOS INDICES ###
#######################################
using DelimitedFiles

dataset_iris = readdlm("iris.data",',');
cat_indexes_iris = 5;
dataset_iris = cat_to_num(dataset_iris, cat_indexes_iris);

dataset_haber = readdlm("haberman.data",',');
cat_indexes_haber = 4;
dataset_haber = cat_to_num(dataset_haber, cat_indexes_haber);
#######################################

m= maximum(dataset_haber[:,1:3], dims=1)
mi=minimum(dataset_haber[:,1:3], dims=1)
using Statistics
media=mean(dataset_haber[:,1:3], dims=1)
sd= std(dataset_haber[:,1:3], dims=1)
info=[m mi media sd]


d=dataset_haber[:,1:3]
# let info=rand(size(d, 2),4)
info=rand(size(d, 2),4)
for i=1:size(d, 2)
    c=dataset_haber[:,i]
    m= maximum(c, dims=1)
    mi=minimum(c, dims=1)
    media=mean(c, dims=1)
    sd= std(c, dims=1)
    info[i,:]=[m mi media sd]
end
info

media=transpose(info[:,3])
sd=transpose(info[:,4])
norm= (d.-media)./sd

maximos=transpose(info[:,1])
minimos=transpose(info[:,2])
norm_2= (d.-minimos)./(maximos.-minimos)
