
#                  PRÁCTICA 2 APRENDIZAJE AUTOMÁTICO I
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


push!(LOAD_PATH, ".")
using Main.modelComparison
using Main.funcionesUtiles
using Images
using JLD2
using Statistics
import Base.display
import FileIO


############################## FUNCIONES ##############################

##### symmetry

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


##### featureExtraction

function featureExtraction(window::Array{Float64}) #la ventana es una imagen pasada a array
    if typeof(window)==Array{Float64, 3}
        car = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            car[1,cnt] = mean(color)
            cnt += 1
            car[1,cnt] = std(color)
            cnt += 1
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        sym=symmetry(window)
        car = zeros(1,6)
        car[1,1] = mean(window)
        car[1,2] = std(window)
        car[1,3]=mean(sym[1])
        car[1,4]=std(sym[1])
        car[1,5]=mean(sym[2])
        car[1,6]=std(sym[2])

        car = hcat(car, differences(window))
    end

    return car
end


##### ClassifyEye

function ClassifyEye(img, rna, threshold, NormalizationParameters, log=false)
    ColorDataset = funcionesUtiles.imageToColorArray(img)
    GrayDataset = funcionesUtiles.imageToGrayArray(img)
    charC = featureExtraction(ColorDataset)
    charG = featureExtraction(GrayDataset)
    char = hcat(charC, charG)
    normalizeZeroMean!(char, NormalizationParameters)
    result = rna(char')
    if log
        print(result)
    end
    if result[1] >= threshold
        return 1
    else
        return 0
    end
end


##### differences

function differences(grayImage)
    grayImage = grayImage/mean(grayImage) #BRILLO
    rows = size(grayImage)[1]
    columns = size(grayImage)[2]
    diff_matrix = zeros(rows, columns)

    for i in 1:rows
        for j in 1:columns-1
            diff = abs(grayImage[i,j]-grayImage[i,j+1])
            diff_matrix[i,j] = diff
        end
    end

    #return diff_matrix

    result = zeros(1,9)

    # TOP
    result[1,1] = mean(diff_matrix[1,:])
    result[1,2] = mean(diff_matrix[2,:])
    result[1,3] = mean(diff_matrix[3,:])

    # MIDDLE
    mid = div(rows,2)
    result[1,4] = mean(diff_matrix[mid-1,:])
    result[1,5] = mean(diff_matrix[mid,:])
    result[1,6] = mean(diff_matrix[mid+1,:])

    # BOTTOM
    result[1,7] = mean(diff_matrix[end-2,:])
    result[1,8] = mean(diff_matrix[end-1,:])
    result[1,9] = mean(diff_matrix[end,:])

    return result

end

"""
img_path = "positivos/image_0001-1.bmp";
img = load(img_path);
display(img)
grayImage = funcionesUtiles.imageToGrayArray(img);
x = differences(grayImage)
display(x)


img_path = "negativos/2.bmp";
img = load(img_path);
display(img)
grayImage = funcionesUtiles.imageToGrayArray(img);
x = differences(grayImage)
display(x)
"""

############################### CÓDIGO ###############################

# Cargamos los dataset y extraemos las características
(colorDataset, grayDataset, targets) = funcionesUtiles.loadTrainingDataset();
inputsC = vcat(featureExtraction.(colorDataset)...);
inputsG = vcat(featureExtraction.(grayDataset)...);
inputs = [inputsC inputsG];
positive = colorDataset[targets .== 1];

# Normalizamos los input y creamos en dataset de entrenamiento
normalizeMinMax!(inputs);
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Entrenamos la red y comprobamos su funcionamiento
(rna,losses) = trainClassANN([4], trainingDataset);
testThreshold = 0.5;

### Positivo
img_path = "positivos/image_0003-1.bmp";
img = load(img_path);
eye = ClassifyEye(img, rna, testThreshold, NormalizationParameters, true)

### Negativo
img_path1 = "negativos/17.bmp";
img1 = load(img_path1);
eye = ClassifyEye(img1, rna, testThreshold, NormalizationParameters, true)


# Entrenamos con CrossValidation
k = 10;
topology = [4];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;

parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
#modelCrossValidation(:ANN, parameters, inputsG, targets, k)


# Utilizamos la red para detectar ojos
img_path = "testFinal/image_0263.jpg";
img = load(img_path);
display(img)

minWindowSizeY = minimum(size.(colorDataset, 1));
maxWindowSizeY = maximum(size.(colorDataset, 1));
minWindowSizeX = minimum(size.(colorDataset, 2));
maxWindowSizeX = maximum(size.(colorDataset, 2));

detectionThreshold = 0.7;
windowLocations = Array{Int64,1}[];
for windowWidth = minWindowSizeX:4:maxWindowSizeX
    for windowHeight = minWindowSizeY:4:maxWindowSizeY
        for x1 = 1:10:size(img,2)-windowWidth
            for y1 = 1:10:size(img,1)-windowHeight
                x2 = x1 + windowWidth;
                y2 = y1 + windowHeight;
                if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters) == 1
                    push!(windowLocations, [x1, x2, y1, y2]);
                end
            end
        end
    end
end

colorImage = funcionesUtiles.imageToColorArray(img2);
for i=1:size(windowLocations)[1]
    funcionesUtiles.setRedBox!(colorImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
end
display(colorImage)
