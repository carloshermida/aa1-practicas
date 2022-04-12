# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------
#push!(LOAD_PATH, ".")
import Main.modelComparison
import Main.funcionesUtiles
using Images
using JLD2
using Statistics
import Base.display
import FileIO

function featureExtraction(window::Array{Float64, 3}) #la ventana es una imagen pasada a array
    car = zeros(2,3)
    for i=1:3
        color = window[:,:,i]
        car[1,i] = mean(color)
        car[2,i] = std(color)
    end
    return car
end;

function featureExtraction(window::Array{Float64, 2})
    car = zeros(2,2)
    for i=1:2
        color = window[:,i]
        car[1,i] = mean(color)
        car[2,i] = std(color)
    end
    return car
end;

(colorDataset, grayDataset, targets) = loadTrainingDataset();
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

img_path = "positivos/image_0003-1.bmp"
img = load(img_path)
GrayDataset = imageToGrayArray(img)
futureExtraction(img)
"""
