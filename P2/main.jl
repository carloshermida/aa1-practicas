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

function ClassifyEye(img, rna, threshold, NormalizationParameters)
    ColorDataset = funcionesUtiles.imageToColorArray(img)
    GrayDataset = funcionesUtiles.imageToGrayArray(img)
    charC = featureExtraction(ColorDataset)
    charG = featureExtraction(GrayDataset)
    char = hcat(charC, charG)
    normalizeZeroMean!(char, NormalizationParameters)
    result = rna(char')
    if result[1] <= threshold
        return 1
    else
        return 0
    end
end;


(colorDataset, grayDataset, targets) = funcionesUtiles.loadTrainingDataset();
inputsC = vcat(featureExtraction.(colorDataset)...);
inputsG = vcat(featureExtraction.(grayDataset)...);
inputs = [inputsC inputsG]

positive = colorDataset[targets .== 1]

inputs = normalizeZeroMean(inputs) ######### datos acotados???
NormalizationParameters = calculateZeroMeanNormalizationParameters(inputs)

targetsMatrix = reshape(targets, length(targets), 1) ############# borrar????
trainingDataset = (inputs, targetsMatrix)

(rna,losses) = trainClassANN([4], trainingDataset)

# COMPROBAR

img_path = "positivos/image_0003-1.bmp"
img = load(img_path);
eye = ClassifyEye(img, rna, 0.01, NormalizationParameters)


img_path1 = "negativos/17.bmp"
img1 = load(img_path1);
eye = ClassifyEye(img1, rna, 0.01, NormalizationParameters)

##########################


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


img_path2 = "testFinal/image_0263.jpg"
img2 = load(img_path2)

ColorDataset = funcionesUtiles.imageToColorArray(img2)
minWindowSizeY = minimum(size.(positive, 1));
maxWindowSizeY = maximum(size.(positive, 1));
minWindowSizeX = minimum(size.(positive, 2));
maxWindowSizeX = maximum(size.(positive, 2));
windowLocations = Array{Int64,1}[];
for windowWidth = minWindowSizeX:4:maxWindowSizeX
    for windowHeight = minWindowSizeY:4:maxWindowSizeY
        for x1 = 1:10:size(img2,2)-windowWidth
            for y1 = 1:10:size(img2,1)-windowHeight
                x2 = x1 + windowWidth;
                y2 = y1 + windowHeight;
                #print(ColorDataset[y1:y2, x1:x2, :])ç
                #print(typeof(img2[y1:y2, x1:x2, :]))
                if ClassifyEye(img2[y1:y2, x1:x2], rna, 0.01, NormalizationParameters) == 1
                    push!(windowLocations, [x1, x2, y1, y2]);
                end;
            end;
        end;
    end;
end;


for i=1:size(windowLocations)[1]
    funcionesUtiles.setRedBox!(ColorDataset, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
end

display(ColorDataset)
