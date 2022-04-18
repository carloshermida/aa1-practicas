# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------
push!(LOAD_PATH, ".")
using Main.modelComparison
using Main.funcionesUtiles
using Images
using JLD2
using Statistics
import Base.display
import FileIO


function symmetry(window::Array{Float64,2})
    rows=size(window,1)
    columns=size(window,2)
    VerticalSymmetrical=zeros(rows, columns)
    HorizontalSymmetrical=zeros(rows, columns)
    for i=1:rows
        for j=1:columns
            VerticalSymmetrical[i,j]=window[i,columns+1-j]
            HorizontalSymmetrical[i,j]=window[rows+1-i, j]
        end
    end
    VError=abs.(window-VerticalSymmetrical)
    HError=abs.(window-HorizontalSymmetrical)
    return (VError, HError)
end

function featureExtraction(window::Array{Float64}) #la ventana es una imagen pasada a array
    if typeof(window)==Array{Float64, 3}
        car = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]
            car[1,cnt] = mean(color)
            cnt += 1
            car[1,cnt] = std(color)
            cnt += 1
        end

    elseif typeof(window)==Array{Float64, 2}
        sym=symmetry(window)
        car = zeros(1,6)
        car[1,1] = mean(window)
        car[1,2] = std(window)
        car[1,3]=mean(sym[1])
        car[1,4]=std(sym[1])
        car[1,5]=mean(sym[2])
        car[1,6]=std(sym[2])
    end

    return car
end;

function ClassifyEye(window, rna, threshold)
    window = funcionesUtiles.imageToGrayArray(window)
    char = featureExtraction(window)'
    result = rna(char)
    if result[1] <= threshold
        return 1
    else
        return 0
    end
end;


(colorDataset, grayDataset, targets) = funcionesUtiles.loadTrainingDataset();
inputsC = vcat(featureExtraction.(colorDataset)...);
inputsG = vcat(featureExtraction.(grayDataset)...);


"""
colorDatasetN, grayDatasetN = loadFolderImages("negativos")
inputsCN = vcat(featureExtraction.(colorDatasetN)...);
inputsGN = vcat(featureExtraction.(grayDatasetN)...);

colorDatasetP, grayDatasetP = loadFolderImages("positivos")
inputsCP = vcat(featureExtraction.(colorDatasetP)...);
inputsGP = vcat(featureExtraction.(grayDatasetP)...);

img_path = "positivos/image_0003-1.bmp"
img = load(img_path)
ColorDataset = imageToColorArray(img)
futureExtraction(img)
"""

img_path = "positivos/image_0003-1.bmp"
img = load(img_path)
GrayDataset = funcionesUtiles.imageToGrayArray(img)
symmetry(GrayDataset)
char= featureExtraction(GrayDataset)'
eye = ClassifyEye(img, rna, 0.3)


img_path1 = "negativos/6.bmp"
img1 = load(img_path1)
GrayDataset1 = funcionesUtiles.imageToGrayArray(img1)
GrayDataset1[:,1]
symmetry(GrayDataset1)
char=featureExtraction(GrayDataset1)'
eye = ClassifyEye(img1, rna, 0.3)

inputsG = normalizeZeroMean(inputsG) ######### datos acotados???
targetsMatrix = reshape(targets, length(targets), 1) ############# borrar????
trainingDataset = (inputsG, targetsMatrix)


k = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4]; # Una capa oculta con 2 neuronas
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

parameters = Dict()
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputsG, targets, k)

(rna,losses) = trainClassANN([4], trainingDataset)
