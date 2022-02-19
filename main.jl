
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
        ann = Chain(ann...,  Dense(numInputsLayer, n_output, identity), softmax);
    end
    return ann
end


### entrenamiento

function entrenar(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01)
    dataset = (Matrix(dataset[1]'),Matrix(dataset[2]')) # Trasponemos las matrices para que los patrones estén en columnas
    n_inputs = size(dataset[1])[1]
    n_outputs = size(dataset[2])[1]
    red = rna(topology, n_inputs, n_outputs) # Creamos la red adecuada a la base de datos
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(red(x),y) : Flux.Losses.crossentropy(red(x),y); # Definimos la función loss en base a nuestra red (copiada del pdf)
    losses = zeros(0) # Vamos a almacenar el loss de cada ciclo de entrenamiento en esta variable
    for _ = 1:100   # Número de ciclos de entrenamiento (lo puse aleatorio). Se regulará en la siguiente práctica
        Flux.train!(loss, params(red), [dataset], ADAM(learningRate));  # Entrenamos la red con la función train! de la libreria FLux (copiado del pdf)
        append!(losses, loss(dataset[1],dataset[2])) # Añadimos el loss de cada ciclo
    end
    return (red, losses) # Devolvemos la red entrenada y el vector de losses
end

function entrenar(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01)
    target = reshape(dataset[2], (length(dataset[2]),1)) # Cambiamos el vector target a una matriz columna. No es necesario crear la columna contraria, porque cuando utilicemos la red entrenada, devolverá una matriz columna con valores entre 0 y 1, y a través del umbral ya decide si pertenece a la clase A o a la clase B
    inputs = dataset[1]                                  # Si tiene dos clases, solo hace falta una neurona de salida, pero si tiene más, harán falta tantas neuronas de salidas como clases.
    dataset = tuple(inputs, target)
    return entrenar(topology, dataset, maxEpochs, minLoss, learningRate) # Ahora ya le pasamos 2 matrices, por lo que va a la función anterior
end


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

##### Entrenamiento de red

### HABERMAN

dataset_haber = readdlm("haberman.data",','); # Cargamos los datos
dataset_haber = dataset_haber'
feature_haber = dataset_haber[4,:]
feature_haber = convert(AbstractArray{<:Any,1},feature_haber);
target = oneHotEncoding(feature_haber)
numerics_haber = dataset_haber[1:3,:]'
input = normalizeMinMax!(numerics_haber)

dataset = (input, target) # Creamos la tupla con los datos (sin trasponer, ya lo hace dentro)
red_entrenada,losses = entrenar([3], dataset) # Entrenamos la red con 1 capa oculta de tres neuronas, durante 100 ciclos (nº arbitrario, se cambiará en la siguiente práctica)
# Esto a simple vista devuelve la misma red, pero si le aplicamos la funcióin params(red) antes y despues del bucle de entrenamiento, observamos que los pesos cambian

output = red_entrenada(input') # Le pasamos a la red entrenada un input (en este caso es el mismo con el qque entrenó), para que nos devuelva la probabilidad de pertenencia a cada clase
target = reshape(target, (length(target),1)) # Convertimos target en una matriz columna para poder llamar a la función accuracy
accuracy(target, Matrix(output')) # Acuraccy nos devuelve el porcentaje de acierto (como output es una matriz columna, convertimos antes target para que entre en el accuracy {bool,2} , {real,2}, porque no hay {bool,1} , {real,2})

### IRIS

dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
feature_iris = dataset_iris[5,:]
feature_iris = convert(AbstractArray{<:Any,1},feature_iris);
target = oneHotEncoding(feature_iris)
numerics_iris = convert(AbstractArray{Float64,2},dataset_iris[1:4,:]')
input = normalizeMinMax!(numerics_iris)

dataset = (input, target)
red_entrenada,losses = entrenar([3], dataset)

output = red_entrenada(input')
accuracy(target, Matrix(output'))




###################################################
# Bucle para comprobar la mejor arquitectura
#####################

# Es un bucle que comprueba la mejor arquitectura para la red suponiendo el número de capas ocultas = 2
# La que obtenga una mejor accuracy es la combinación óptima. En este caso probamos entre 1 y 4 neuronas por capa

# Calculamos la media de 10 entrenamientos por arquitectura, pues no siempre obtemos la misma precisión

"""
let
    best = 0
    match = 0
    for i = 1:4
        for j = 1:4
            x = zeros(0)
            for _ = 1:10
                dataset = (input, target)
                red_entrenada,losses = entrenar([i,j], dataset)
                output = red_entrenada(input')
                append!(x, accuracy(target, Matrix(output')))
            end
            result = mean(x)
            print(i, "\t", j, "\t", result, "\n")
            if result > best
                best = result
                match = (i, j, best)
            end
        end
    end
    print("BEST: ", match[1], "\t", match[2], "\t", match[3])
end
"""
