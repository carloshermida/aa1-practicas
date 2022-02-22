
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.3
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


using Statistics
using DelimitedFiles
using Flux
using Random

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
    norm = (numerics.-MinMax[2])./(MinMax[1].-MinMax[2])
    for i=1:size(numerics,2)
        if MinMax[1][i]==MinMax[2][i]
            norm[:,i].=0
        end
    end
    return norm
end

# Función sobrecargada 2
normalizeMinMax!(numerics::AbstractArray{<:Real,2})=(normalizeMinMax!(numerics, calculateMinMaxNormalizationParameters(numerics)))

# Función sobrecargada 3
function normalizeMinMax(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    norm = (copia.-MinMax[2])./(MinMax[1].-MinMax[2])
    for i=1:size(copia,2)
        if MinMax[1][i]==MinMax[2][i]
            norm[:,i].=0
        end
    end
    return norm
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
    norm = (numerics.-media_sd[1])./media_sd[2]
    for i=1:size(numerics,2)
        if media_sd[2][i]==0
            norm[:,i].=0
        end
    end

    return norm
end

# Función sobrecargada 2
normalizeZeroMean!(numerics::AbstractArray{<:Real,2})=(normalizeZeroMean!(numerics,calculateZeroMeanNormalizationParameters(numerics_haber)))

# Función sobrecargada 3
function normalizeZeroMean(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    copia=copy(numerics)
    norm = (copia.-media_sd[1])./media_sd[2]
    for i=1:size(copia,2)
        if media_sd[2][i]==0
            norm[:,i].=0
        end
    end
    return norm
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


##### red neuronal artificial

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


##### entrenamiento

# Función principal
function entrenar(;topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01, validacion::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)),
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)), maxEpochsVal::Int = 20)

    dataset = (Matrix(dataset[1]'),Matrix(dataset[2]')) # Trasponemos las matrices para que los patrones estén en columnas
    n_inputs = size(dataset[1])[1]
    n_outputs = size(dataset[2])[1]
    red = rna(topology, n_inputs, n_outputs)
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(red(x),y) : Flux.Losses.crossentropy(red(x),y);
    losses = zeros(0)
    cnt = 0
    best_loss = 100
    for i = 1:maxEpochs
        Flux.train!(loss, params(red), [dataset], ADAM(learningRate))
        if validacion != tuple(zeros(0,0), falses(0,0))
            local_loss = loss(Matrix(validacion[1]'), Matrix(validacion[2]'))
            if local_loss < best_loss
                best_red = deepcopy(red)
                best_loss = local_loss
                cnt = 0
            else
                cnt += 1
            end
            push!(losses, local_loss)
            if cnt == maxEpochsVal
                return (best_red, losses)
            end
        else
            push!(losses, loss(dataset[1],dataset[2]))
        end
    end
    return (red, losses)
end

# Función sobrecargada
function entrenar(; topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01, test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=tuple(zeros(0,0), falses(0)),
    validacion::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=tuple(zeros(0,0), falses(0)), maxEpochsVal::Int = 20)

    target = reshape(dataset[2], (length(dataset[2]),1))
    target_test = reshape(test[2], (length(test[2]),1))
    target_val = reshape(validacion[2], (length(validacion[2]),1))  # Cambiamos el vector target a una matriz columna. No es necesario crear la columna contraria, porque cuando utilicemos la red entrenada, devolverá una matriz columna con valores entre 0 y 1, y a través del umbral ya decide si pertenece a la clase A o a la clase B
    inputs = dataset[1]
    inputs_test = test[1]
    inputs_val = validacion[1]                                # Si tiene dos clases, solo hace falta una neurona de salida, pero si tiene más, harán falta tantas neuronas de salidas como clases.
    dataset = tuple(inputs, target)
    test = tuple(inputs_test, target_test)
    validacion = tuple(inputs_val, target_val)
    return entrenar(topology=topology, dataset=dataset, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, 
                    test=test, validacion=validacion, maxEpochsVal=maxEpochsVal) # Ahora ya le pasamos 2 matrices, por lo que va a la función anterior
end


##### sobreentrenamiento

# Función para conjunto de entrenamiento y test
function holdOut(N, P)
    d = randperm(N)
    test = d[1:Integer(round(P*N))]
    entrenamiento = d[Integer(round(P*N))+1:end]
    return (entrenamiento, test)
end

# Función para conjunto de entrenamiento, validación y test
function holdOut(N, Pval, Ptest)
    d, test = holdOut(N, Ptest)
    entrenamiento, validacion = holdOut(size(d)[1], Pval)
    entrenamiento = d[entrenamiento]
    validacion = d[validacion]
    return (entrenamiento, validacion, test)
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


##### Entrenamiento de red

### HABERMAN

dataset_haber = readdlm("haberman.data",','); # Cargamos los datos
dataset_haber = dataset_haber'
feature_haber = dataset_haber[4,:]
feature_haber = convert(AbstractArray{<:Any,1},feature_haber);
target = oneHotEncoding(feature_haber)
numerics_haber = dataset_haber[1:3,:]'
input = normalizeMinMax!(numerics_haber)

# Dividimos los datos en tres conjuntos
train_h, val_h, test_h = holdOut(size(input)[1], 0.1, 0.3)
train = (input[train_h, 1:3], target[train_h])
val = (input[val_h, 1:3], target[val_h])
test = (input[test_h, 1:3], target[test_h])

red_entrenada,losses = entrenar(topology=[3], dataset=train, validacion=val) # Entrenamos la red con 1 capa oculta de tres neuronas, durante 100 ciclos (nº arbitrario, se cambiará en la siguiente práctica)
# Esto a simple vista devuelve la misma red, pero si le aplicamos la funcióin params(red) antes y despues del bucle de entrenamiento, observamos que los pesos cambian

# Obtenemos la precisión con el conjunto de test
output = red_entrenada(test[1]')
accuracy(test[2], Matrix(output'))

### IRIS

# Cargamos los datos
dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
feature_iris = dataset_iris[5,:]
feature_iris = convert(AbstractArray{<:Any,1},feature_iris);
target = oneHotEncoding(feature_iris)
numerics_iris = convert(AbstractArray{Float64,2},dataset_iris[1:4,:]')
input = normalizeZeroMean(numerics_iris, calculateZeroMeanNormalizationParameters(numerics_iris))

# Dividimos los datos en tres conjuntos
train_i, val_i, test_i = holdOut(size(input)[1], 0.1, 0.3)
train = (input[train_i, 1:4], target[train_i, 1:3])
val = (input[val_i, 1:4], target[val_i, 1:3])
test = (input[test_i, 1:4], target[test_i, 1:3])

# Entrenamos la red con el conjunto de entrenamiento y validación
red_entrenada,losses = entrenar(topology = [3], dataset = train, validacion = val)

# Obtenemos la precisión con el conjunto de test
output = red_entrenada(test[1]')
accuracy(test[2], Matrix(output'))


### IRIS (sin normalizar)

dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
feature_iris = dataset_iris[5,:]
feature_iris = convert(AbstractArray{<:Any,1},feature_iris);
target = oneHotEncoding(feature_iris)
input = convert(AbstractArray{Float64,2},dataset_iris[1:4,:]')

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
