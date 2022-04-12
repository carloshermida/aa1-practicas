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
import Math

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
        car = zeros(2,3)
        for i=1:3
            color = window[:,:,i]
            car[1,i] = mean(color)
            car[2,i] = std(color)
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
featureExtraction(GrayDataset)

img_path1 = "negativos/7.bmp"
img1 = load(img_path1)
GrayDataset1 = funcionesUtiles.imageToGrayArray(img1)
GrayDataset1[:,1]
symmetry(GrayDataset1)
featureExtraction(GrayDataset1)
