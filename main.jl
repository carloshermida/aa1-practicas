
#               PRÁCTICA APRENDIZAJE AUTOMÁTICO I / v1.3
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


using Statistics
using DelimitedFiles
using Flux
using Random
using Plots

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
function entrenar(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01, validacion::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)),
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)), maxEpochsVal::Int = 20)

    dataset = (Matrix(dataset[1]'),Matrix(dataset[2]')) # Trasponemos las matrices para que los patrones estén en columnas
    n_inputs = size(dataset[1])[1]
    n_outputs = size(dataset[2])[1]
    red = rna(topology, n_inputs, n_outputs)
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(red(x),y) : Flux.Losses.crossentropy(red(x),y);
    losses_train = zeros(0)
    losses_val = zeros(0)
    losses_test = zeros(0)
    cnt = 0
    best_loss = 100
    best_red = 0
    acc = 404

    push!(losses_train, loss(dataset[1],dataset[2]))
    if test != tuple(zeros(0,0), falses(0,0))
        push!(losses_test, loss(Matrix(test[1]'), Matrix(test[2]')))
    end
    if validacion != tuple(zeros(0,0), falses(0,0))
        push!(losses_val,loss(Matrix(validacion[1]'), Matrix(validacion[2]')))
    end

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

            push!(losses_train, loss(dataset[1],dataset[2]))
            if test != tuple(zeros(0,0), falses(0,0))
                push!(losses_test, loss(Matrix(test[1]'), Matrix(test[2]')))
            end
            push!(losses_val, local_loss)

            if cnt == maxEpochsVal
                if test != tuple(zeros(0,0), falses(0,0))
                    output = best_red(test[1]')
                    acc = accuracy(test[2], Matrix(output'))
                end
                return (best_red, (losses_train,losses_val,losses_test), acc)
            end
        else
            push!(losses_train, loss(dataset[1],dataset[2]))
            if test != tuple(zeros(0,0), falses(0,0))
                push!(losses_test, loss(Matrix(test[1]'), Matrix(test[2]')))
            end
        end

    end

    if test != tuple(zeros(0,0), falses(0,0))
        output = red(test[1]')
        acc = accuracy(test[2], Matrix(output'))
    end
    return (red, (losses_train,losses_val,losses_test), acc)
end


# Función sobrecargada
function entrenar(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01, test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=tuple(zeros(0,0), falses(0)),
    validacion::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=tuple(zeros(0,0), falses(0)), maxEpochsVal::Int = 20)

    if validacion != tuple(zeros(0,0), falses(0))
        target_val = reshape(validacion[2], (length(validacion[2]),1))
        inputs_val = validacion[1]
        validacion = tuple(inputs_val, target_val)
    else
        validacion = tuple(zeros(0,0), falses(0,0))
    end

    if test != tuple(zeros(0,0), falses(0))
        target_test = reshape(test[2], (length(test[2]),1))
        inputs_test = test[1]
        test = tuple(inputs_test, target_test)
    else
        test = tuple(zeros(0,0), falses(0,0))
    end

    target = reshape(dataset[2], (length(dataset[2]),1))
    inputs = dataset[1]                               # Si tiene dos clases, solo hace falta una neurona de salida, pero si tiene más, harán falta tantas neuronas de salidas como clases.
    train = tuple(inputs, target)

    return entrenar(topology, train, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
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


##### matriz de confusión

# Funcion principal
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    VN = sum(outputs .== targets .== 0)
    VP = sum(outputs .== targets .== 1)
    FN = sum(outputs .== 0 .!= targets)
    FP = sum(outputs .== 1 .!= targets)

    acc = (VN + VP)/(VN + VP + FN + FP)
    error_rate = (FN + FP)/(VN + VP + FN + FP)

    if VN == length(outputs)
        sensivity = 1
        pos_pred_val = 1
    else
        sensivity = VP/(FN+VP)
        pos_pred_val =  VP/(VP+FP)
    end

    if VP == length(outputs)
        specificity = 1
        neg_pred_val = 1
    else
        specificity = VN/(FP+VN)
        neg_pred_val = VN/(VN+FN)
    end

    if pos_pred_val == sensivity == 0
        f1_score = 0
    else
        f1_score = (2 * sensivity * pos_pred_val) / (sensivity + pos_pred_val)
    end

    conf_matrix = reshape([VN, FN, FP, VP], (2,2))

    metrics = [acc, error_rate, sensivity, specificity, pos_pred_val, neg_pred_val, f1_score, conf_matrix]
    replace!(metrics, NaN => 0)
    return metrics
end


# Función sobrecargada
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold)
    outputs = outputs .>= threshold
    return confusionMatrix(outputs, targets)
end



##
############################### CÓDIGO ###############################

##### HABERMAN

dataset_haber = readdlm("haberman.data",',');
dataset_haber = dataset_haber'
feature_haber = dataset_haber[4,:]
feature_haber = convert(AbstractArray{<:Any,1},feature_haber);
target = oneHotEncoding(feature_haber)

input = dataset_haber[1:3,:]'

# Dividimos los datos en tres conjuntos
train_h, val_h, test_h = holdOut(size(input)[1], 0.3, 0.2)

input_train = input[train_h, 1:3]
maxmin_train = calculateMinMaxNormalizationParameters(input_train)
input_train = normalizeMinMax(input_train, maxmin_train)
train = (input_train, target[train_h])

input_val = normalizeMinMax(input[val_h, 1:3], maxmin_train)
val = (input_val, target[val_h])

input_test = normalizeMinMax(input[test_h, 1:3], maxmin_train)
test = (input_test, target[test_h])

red_entrenada,losses, acc = entrenar([3], train, validacion = val, test = test)

g = plot()
plot!(g,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(g,1:length(losses[2]), losses[2], label = "validación")
plot!(g,1:length(losses[3]), losses[3], label = "test")

red_entrenada,losses, acc = entrenar([2,2], train,validacion = val, test = test)
h = plot()
plot!(h,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(h,1:length(losses[2]), losses[2], label = "validación")
plot!(h,1:length(losses[3]), losses[3], label = "test")

red_entrenada,losses, acc = entrenar([8], train,validacion = val, test = test)
k = plot()
plot!(k,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(k,1:length(losses[2]), losses[2], label = "validación")
plot!(k,1:length(losses[3]), losses[3], label = "test")

plot(g,h,k,layout = (3,1))


# Matriz de confusión
outputs = red_entrenada(test[1]')[1,:]
targets = test[2]
metrics = confusionMatrix(outputs, targets, 0.5)


##### IRIS

# Cargamos los datos
dataset_iris = readdlm("iris.data",',');
dataset_iris = permutedims(dataset_iris)
feature_iris = dataset_iris[5,:]
feature_iris = convert(AbstractArray{<:Any,1},feature_iris);
target = oneHotEncoding(feature_iris)
numerics_iris = convert(AbstractArray{Float64,2},dataset_iris[1:4,:]')
input = normalizeZeroMean(numerics_iris, calculateZeroMeanNormalizationParameters(numerics_iris))

# Dividimos los datos en tres conjuntos
train_i, val_i, test_i = holdOut(size(input)[1], 0.2, 0.1)
train = (input[train_i, 1:4], target[train_i, 1:3])
val = (input[val_i, 1:4], target[val_i, 1:3])
test = (input[test_i, 1:4], target[test_i, 1:3])

# Entrenamos la red con el conjunto de entrenamiento y validación
red_entrenada,losses, acc = entrenar([4], train,validacion = val, test = test)
g = plot()
plot!(g,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(g,1:length(losses[2]), losses[2], label = "validación")
plot!(g,1:length(losses[3]), losses[3], label = "test")

red_entrenada,losses, acc = entrenar([4,4], train,validacion = val, test = test)
h = plot()
plot!(h,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(h,1:length(losses[2]), losses[2], label = "validación")
plot!(h,1:length(losses[3]), losses[3], label = "test")

red_entrenada,losses, acc = entrenar([8], train,validacion = val, test = test)
k = plot()
plot!(k,1:length(losses[1]), losses[1], label = "entrenamiento")
plot!(k,1:length(losses[2]), losses[2], label = "validación")
plot!(k,1:length(losses[3]), losses[3], label = "test")

plot(g,h,k,layout = (3,1))



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
