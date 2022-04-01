
#                  PRÁCTICA 1 APRENDIZAJE AUTOMÁTICO I
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


using Statistics
using DelimitedFiles
using Flux
using Random
using Plots
using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 1 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
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
end;

# Función sobrecargada 1
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1})=oneHotEncoding(feature::AbstractArray{<:Any,1}, unique(feature));

# Función sobrecargada 2
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado
oneHotEncoding(feature::AbstractArray{Bool,1}) = feature;
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

##### Máximos y mínimos
# Funciones para calcular los parametros de normalizacion y normalizar
# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
# Función auxiliar
function calculateMinMaxNormalizationParameters(numerics::AbstractArray{<:Real,2})
    m= maximum(numerics, dims=1)
    mi=minimum(numerics, dims=1)
    return (m, mi)
end;

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)


# Función sobrecargada 1
function normalizeMinMax!(numerics::AbstractArray{<:Real,2}, MinMax::NTuple{2, AbstractArray{<:Real,2}})
    norm = (numerics.-MinMax[2])./(MinMax[1].-MinMax[2])
    for i=1:size(numerics,2)
        if MinMax[1][i]==MinMax[2][i]
            norm[:,i].=0
        end
    end
    return norm
end;

# Función sobrecargada 2
normalizeMinMax!(numerics::AbstractArray{<:Real,2})=(normalizeMinMax!(numerics, calculateMinMaxNormalizationParameters(numerics)));

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
end;

# Función sobrecargada 4
normalizeMinMax(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeMinMax(copia, calculateMinMaxNormalizationParameters(numerics)));


##### Media y desviación típica

# Función principal
function calculateZeroMeanNormalizationParameters(numerics::AbstractArray{<:Real,2})
    media=mean(numerics, dims=1)
    sd= std(numerics, dims=1)
    return (media, sd)
end;

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)

# Función sobrecargada 1
function normalizeZeroMean!(numerics::AbstractArray{<:Real,2}, media_sd::NTuple{2, AbstractArray{<:Real,2}})
    norm = (numerics.-media_sd[1])./media_sd[2]
    for i=1:size(numerics,2)
        if media_sd[2][i]==0
            norm[:,i].=0
        end
    end

    return norm
end;

# Función sobrecargada 2
normalizeZeroMean!(numerics::AbstractArray{<:Real,2})=(normalizeZeroMean!(numerics,calculateZeroMeanNormalizationParameters(numerics)));

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
end;

# Función sobrecargada 4
normalizeZeroMean(numerics::AbstractArray{<:Real,2})=(copia=copy(numerics);normalizeZeroMean(copia,calculateZeroMeanNormalizationParameters(numerics)));


# ClassifyOutputs
# Funciones auxiliar que permite transformar una matriz de
#  valores reales con las salidas del clasificador o clasificadores
#  en una matriz de valores booleanos con la clase en la que sera clasificada
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
end;


# accuracy
# Funciones para calcular la precision
# Función sobrecargada 1
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1}) = mean(outputs.==targets);

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
        return mean(classComparison)
    end
end;

# Función sobrecargada 3
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Float64,1}, threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);

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
end;

# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o Float64
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Float32,1}, threshold::Float64=0.5) = accuracy(targets, Float64.(outputs), threshold);

# Funciones para crear y entrenar una RNA
# red neuronal artificial
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
end;


# entrenamiento
function entrenar(topology::AbstractArray{<:Int,1}, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01, validacion::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)),
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=tuple(zeros(0,0), falses(0,0)), maxEpochsVal::Int = 20)

    inputs, targets = Matrix(inputs'), Matrix(targets')
    n_inputs = size(inputs)[1]
    n_outputs = size(targets)[1]
    red = rna(topology, n_inputs, n_outputs)
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(red(x),y) : Flux.Losses.crossentropy(red(x),y);
    losses_train = zeros(0)
    losses_val = zeros(0)
    losses_test = zeros(0)
    cnt = 0
    best_loss = 100
    best_red = 0
    acc = 404

    push!(losses_train, loss(inputs,targets))
    if test != tuple(zeros(0,0), falses(0,0))
        push!(losses_test, loss(Matrix(test[1]'), Matrix(test[2]')))
    end
    if validacion != tuple(zeros(0,0), falses(0,0))
        push!(losses_val,loss(Matrix(validacion[1]'), Matrix(validacion[2]')))
    end

    for i = 1:maxEpochs
        Flux.train!(loss, params(red), [(inputs, targets)], ADAM(learningRate))
        if validacion != tuple(zeros(0,0), falses(0,0))
            local_loss = loss(Matrix(validacion[1]'), Matrix(validacion[2]'))
            if local_loss < best_loss
                best_red = deepcopy(red)
                best_loss = local_loss
                cnt = 0
            else
                cnt += 1
            end

            push!(losses_train, loss(inputs,targets))
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
            push!(losses_train, loss(inputs,targets))
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
function entrenar(topology::AbstractArray{<:Int,1}, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,1};
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

    target = reshape(targets, (length(targets),1))
    train = tuple(inputs, target)

    return entrenar(topology, train, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                    test=test, validacion=validacion, maxEpochsVal=maxEpochsVal)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# sobreentrenamiento
# Función para conjunto de entrenamiento y test
function holdOut(N::Int, P::Float64)
    d = randperm(N)
    test = d[1:Integer(round(P*N))]
    entrenamiento = d[Integer(round(P*N))+1:end]
    return (entrenamiento, test)
end;

# Función para conjunto de entrenamiento, validación y test
function holdOut(N, Pval, Ptest)
    d, test = holdOut(N, Ptest)
    entrenamiento, validacion = holdOut(size(d)[1], Pval)
    entrenamiento = d[entrenamiento]
    validacion = d[validacion]
    return (entrenamiento, validacion, test)
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# Matriz de confusión
# Funcion principal
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    VN = sum(outputs .== targets .== 0)
    VP = sum(outputs .== targets .== 1)
    FN = sum(outputs .== 0 .!= targets)
    FP = sum(outputs .== 1 .!= targets)

    acc = (VN + VP)/(VN + VP + FN + FP)
    error_rate = (FN + FP)/(VN + VP + FN + FP)

    if VN == length(outputs)
        sensitivity = 1
        VPP = 1
    else
        sensitivity = VP/(FN+VP)
        VPP =  VP/(VP+FP)
    end

    if VP == length(outputs)
        specificity = 1
        VPN = 1
    else
        specificity = VN/(FP+VN)
        VPN = VN/(VN+FN)
    end

    if VPP == sensitivity == 0
        f1_score = 0
    else
        f1_score = (2 * sensitivity * VPP) / (sensitivity + VPP)
    end

    conf_matrix = reshape([VN, FN, FP, VP], (2,2))

    metrics = [acc, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_matrix]
    replace!(metrics, NaN => 0)
    return metrics
end

# Función sobrecargada 1
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold=0.5)
    outputs = outputs .>= threshold
    return confusionMatrix(outputs, targets)
end

# Función sobrecargada 2
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, combination = "macro")
    if size(outputs)[2] == size(targets)[2] > 2
        sensitivity = zeros(0)
        specificity = zeros(0)
        VPP = zeros(0)
        VPN = zeros(0)
        f1_score = zeros(0)
        names_metrics = [sensitivity, specificity, VPP, VPN, f1_score]
        numClasses = size(outputs)[2]
        conf_matrix = zeros(numClasses,numClasses)

        for numClass in 1:numClasses
            for numClass2 in 1:numClasses
                conf_matrix[numClass, numClass2] = sum(outputs[:,numClass2] .== targets[:,numClass] .== 1)
            end

            for i in 1:5
                append!(names_metrics[i], confusionMatrix(outputs[:,numClass], targets[:,numClass])[i+2])
            end
        end

    else
        outputs, targets = outputs[:,1], targets[:,1]
        return confusionMatrix(outputs, targets)
    end

    means_metrics = zeros(0)
    if combination == "macro"
        for i in names_metrics
            append!(means_metrics, mean(i))
        end

    elseif combination == "weighted"
        weights = zeros(0)
        for numClass in 1:numClasses
            append!(weights, sum(targets[:,numClass])/size(targets)[1])
        end
        for i in names_metrics
            append!(means_metrics ,sum(i .* weights))
        end
    end

    return (conf_matrix, names_metrics, means_metrics, accuracy(targets, outputs), 1-accuracy(targets, outputs))
end

# Función sobrecargada 3
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, combination = "macro")
    return confusionMatrix(classifyOutputs(outputs), targets, combination)
end

# Función sobrecargada 4
function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}, combination = "macro")
    if size(outputs)[1] == size(targets)[1]
        @assert(all([in(output, unique(targets)) for output in outputs]))
        targets_classes = unique(targets)

        if typeof(outputs) == BitMatrix    ## Si el modelo extrañamente devuelve una matriz columna en vez de vector, la cambiamos por vector
            outputs = outputs[:,1]
        end

        outputs = oneHotEncoding(outputs)
        targets = oneHotEncoding(targets, targets_classes)

        if typeof(targets) == typeof(outputs) == BitVector
            return confusionMatrix(outputs, targets)                ## Si targets y outputs son vectores, van directamente a la confusion matrix de vectores, si no se estará llamando a si misma en bucle
        else
            return confusionMatrix(outputs, targets, combination)
        end
    end
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# Estrategia "uno contra todos"
function crossvalidation(N::Int, k::Int)
    subset_index=collect(1:k)
    rep_subset=repeat(subset_index, Integer(ceil(N/k)))
    rep_subset=rep_subset[1:N]
    return shuffle!(rep_subset)
end

function crossvalidation(targets::AbstractArray{Bool,2}, k)
    subset_index=collect(1:size(targets)[1])
    for class in 1:size(targets)[2]
        class_elements = sum(targets[:,class])
        if class_elements>=k
            estratos=crossvalidation(class_elements, k)
            cnt=1
            for i in 1:size(targets)[1]
                if targets[i,class]==1
                    subset_index[i]=estratos[cnt]
                    cnt+=1
                end
            end
        end
    end
    return subset_index
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k)
    targets=oneHotEncoding(targets)
    return crossvalidation(targets, k)
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

function modelCrossValidation(modelType::Symbol, parameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, k::Int)
    @assert(size(inputs,1)==length(targets));
    classes = unique(targets);
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    crossValidationIndices = crossvalidation(size(inputs,1), k);
    testAcc = Array{Float64,1}(undef, k);
    testF1 = Array{Float64,1}(undef, k);

    for fold in 1:k
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
            trainInputs = inputs[crossValidationIndices.!=fold,:];
            testInputs = inputs[crossValidationIndices.==fold,:];
            trainTargets = targets[crossValidationIndices.!=fold];
            testTargets = targets[crossValidationIndices.==fold];

            if modelType==:SVM
                model = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=parameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(parameters["numNeighbors"]);
            end;

            model = fit!(model, trainInputs, trainTargets);
            testOutputs = predict(model, testInputs);
            if length(unique(testTargets)) > 2
                acc = confusionMatrix(testOutputs, testTargets)[4];
                F1 = confusionMatrix(testOutputs, testTargets)[3][5];
            else
                (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
            end

        else
            @assert(modelType==:ANN);
            trainInputs = inputs[crossValidationIndices.!=fold,:];
            testInputs = inputs[crossValidationIndices.==fold,:];
            trainTargets = targets[crossValidationIndices.!=fold,:];
            testTargets = targets[crossValidationIndices.==fold,:];

            testAccRep = Array{Float64,1}(undef, parameters["numExecutions"]);
            testF1Rep = Array{Float64,1}(undef, parameters["numExecutions"]);

            for numTrain in 1:parameters["numExecutions"]
                if parameters["validationRatio"]>0
                    (trainIndices, validationIndices) = holdOut(size(trainInputs,1), parameters["validationRatio"]*size(trainInputs,1)/size(inputs,1));
                    ann, = entrenar(parameters["topology"],
                        trainInputs[trainIndices,:],   trainTargets[trainIndices,:], maxEpochs = parameters["maxEpochs"],
                        minLoss = 0, learningRate = parameters["learningRate"],
                        validacion=tuple(trainInputs[validationIndices,:], trainTargets[validationIndices,:]),
                        test=tuple(testInputs, testTargets);
                        maxEpochsVal = parameters["maxEpochsVal"]);
                else
                    ann, = entrenar(parameters["topology"],
                        trainInputs, trainTargets, maxEpochs = parameters["maxEpochs"], minLoss = 0,
                        learningRate = parameters["learningRate"], validacion=tuple(zeros(0,0), falses(0,0)),
                        test=tuple(testInputs, testTargets));
                end;

                if size(testTargets)[2] > 1
                    testAccRep[numTrain] = confusionMatrix(collect(ann(testInputs')'), testTargets)[4];
                    testF1Rep[numTrain] = confusionMatrix(collect(ann(testInputs')'), testTargets)[3][5];
                else
                    (testAccRep[numTrain], _, _, _, _, _, testF1Rep[numTrain], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);
                end
            end;
            acc = mean(testAccRep);
            F1  = mean(testF1Rep);
        end;
        testAcc[fold] = acc;
        testF1[fold] = F1;
        println("Results in test in fold ", fold, "/", k, ": accuracy: ", 100*testAcc[fold], " %, F1: ", 100*testF1[fold], " %");
    end;
    println(modelType, ": Average test accuracy on a ", k, "-fold crossvalidation: ", 100*mean(testAcc), ", with a standard deviation of ", 100*std(testAcc));
    println(modelType, ": Average test F1 on a ", k, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    return (testAcc, testF1);
end;


##

############################     IRIS     ############################

Random.seed!(1);

k = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 4;

# Parapetros de kNN
numNeighbors = 3;

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = dataset[:,5];

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Creamos la lista para las gráficas
plot_data = Vector{Any}(zeros(0));

# Entrenamos las RR.NN.AA.
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
push!(plot_data, modelCrossValidation(:ANN, parameters, inputs, targets, k));

# Entrenamos las SVM
parameters = Dict();
parameters["kernel"] = kernel;
parameters["kernelDegree"] = kernelDegree;
parameters["kernelGamma"] = kernelGamma;
parameters["C"] = C;
push!(plot_data, modelCrossValidation(:SVM, parameters, inputs, targets, k));

# Entrenamos los arboles de decision
push!(plot_data, modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, k));

# Entrenamos los kNN
push!(plot_data, modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, k));

# Mostramos las gráficas
p1 = plot(title = "ANN");
plot!(p1, 1:k, plot_data[1][1], label = "accuracy");
plot!(p1, 1:k, plot_data[1][2], label = "f1");
plot!(p1, 1:k, fill(mean(plot_data[1][1]), k), label = "media accuracy");
plot!(p1, 1:k, fill(mean(plot_data[1][2]), k), label = "media f1");

p2 = plot(title = "SVM");
plot!(p2, 1:k, plot_data[2][1], label = "accuracy");
plot!(p2, 1:k, plot_data[2][2], label = "f1");
plot!(p2, 1:k, fill(mean(plot_data[2][1]), k), label = "media accuracy");
plot!(p2, 1:k, fill(mean(plot_data[2][2]), k), label = "media f1");

p3 = plot(title = "DecisionTree");
plot!(p3, 1:k, plot_data[3][1], label = "accuracy");
plot!(p3, 1:k, plot_data[3][2], label = "f1");
plot!(p3, 1:k, fill(mean(plot_data[3][1]), k), label = "media accuracy");
plot!(p3, 1:k, fill(mean(plot_data[3][2]), k), label = "media f1");

p4 = plot(title = "kNN");
plot!(p4, 1:k, plot_data[4][1], label = "accuracy");
plot!(p4, 1:k, plot_data[4][2], label = "f1");
plot!(p4, 1:k, fill(mean(plot_data[4][1]), k), label = "media accuracy");
plot!(p4, 1:k, fill(mean(plot_data[4][2]), k), label = "media f1");

plot(p1, p2, p3, p4, size = (1920, 1080), legend=:bottomleft)




############################     HABERMAN     ############################

Random.seed!(1);

k = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 4;

# Parapetros de kNN
numNeighbors = 3;

# Cargamos el dataset
dataset = readdlm("haberman.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:3]);
targets = dataset[:,4];

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Creamos la lista para las gráficas
plot_data = Vector{Any}(zeros(0));

# Entrenamos las RR.NN.AA.
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
push!(plot_data, modelCrossValidation(:ANN, parameters, inputs, targets, k));

# Entrenamos las SVM
parameters = Dict();
parameters["kernel"] = kernel;
parameters["kernelDegree"] = kernelDegree;
parameters["kernelGamma"] = kernelGamma;
parameters["C"] = C;
push!(plot_data, modelCrossValidation(:SVM, parameters, inputs, targets, k));

# Entrenamos los arboles de decision
push!(plot_data, modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, k));

# Entrenamos los kNN
push!(plot_data, modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, k));

# Mostramos las gráficas
p1 = plot(title = "ANN");
plot!(p1, 1:k, plot_data[1][1], label = "accuracy");
plot!(p1, 1:k, plot_data[1][2], label = "f1");
plot!(p1, 1:k, fill(mean(plot_data[1][1]), k), label = "media accuracy");
plot!(p1, 1:k, fill(mean(plot_data[1][2]), k), label = "media f1");

p2 = plot(title = "SVM");
plot!(p2, 1:k, plot_data[2][1], label = "accuracy");
plot!(p2, 1:k, plot_data[2][2], label = "f1");
plot!(p2, 1:k, fill(mean(plot_data[2][1]), k), label = "media accuracy");
plot!(p2, 1:k, fill(mean(plot_data[2][2]), k), label = "media f1");

p3 = plot(title = "DecisionTree");
plot!(p3, 1:k, plot_data[3][1], label = "accuracy");
plot!(p3, 1:k, plot_data[3][2], label = "f1");
plot!(p3, 1:k, fill(mean(plot_data[3][1]), k), label = "media accuracy");
plot!(p3, 1:k, fill(mean(plot_data[3][2]), k), label = "media f1");

p4 = plot(title = "kNN");
plot!(p4, 1:k, plot_data[4][1], label = "accuracy");
plot!(p4, 1:k, plot_data[4][2], label = "f1");
plot!(p4, 1:k, fill(mean(plot_data[4][1]), k), label = "media accuracy");
plot!(p4, 1:k, fill(mean(plot_data[4][2]), k), label = "media f1");

plot(p1, p2, p3, p4, size = (1920, 1080), legend=:bottomleft)


# NORMALIZAMOS LOS DATOS COMO INDICAMOS EN LA MEMORIA
dataset = readdlm("haberman.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:3])
targets = dataset[:,4];
año = reshape(inputs[:,2], (size(dataset)[1], 1))
inputs[:,2] = normalizeMinMax(año)
edad = reshape(inputs[:,1], (size(dataset)[1], 1))
ganglios = reshape(inputs[:,3], (size(dataset)[1], 1))
inputs[:,1] = normalizeZeroMean(edad)
inputs[:,3] = normalizeZeroMean(ganglios)
