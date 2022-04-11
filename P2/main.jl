# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------
#push!(LOAD_PATH, ".")
#import p1_solution
using Images
using JLD2
using Statistics
import Base.display
import FileIO

imageToGrayArray(image:: Array{<:Colorant,2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
function imageToColorArray(image::Array{<:Colorant,2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    image = RGB.(image);
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;

display(image::Array{Float64,2}) = display(Gray.(image));
display(image::Array{Float64,3}) = (@assert(size(image,3)==3); display(RGB.(image[:,:,1], image[:,:,2], image[:,:,3])); )

function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG", ".BMP"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            # @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            @assert(isa(image, Array{<:Colorant,2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images));
end;

function loadTrainingDataset()
    (positivesColor, positivesGray) = loadFolderImages("positivos");
    (negativesColor, negativesGray) = loadFolderImages("negativos");
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
end;
loadTestDataset() = ((colorMatrix,_) = loadFolderImages("testFinal"); return colorMatrix; );

function setRedBox!(testImage::Array{Float64,3}, minx::Int64, maxx::Int64, miny::Int64, maxy::Int64)
    @assert(size(testImage,3)==3);
    @assert((minx<=size(testImage,2)) && (maxx<=size(testImage,2)) && (miny<=size(testImage,1)) && (maxy<=size(testImage,1)));
    @assert((minx>0) && (maxx>0) && (miny>0) && (maxy>0));
    testImage[miny, minx:maxx, 1] .= 1.;
    testImage[miny, minx:maxx, 2] .= 0.;
    testImage[miny, minx:maxx, 3] .= 0.;
    testImage[maxy, minx:maxx, 1] .= 1.;
    testImage[maxy, minx:maxx, 2] .= 0.;
    testImage[maxy, minx:maxx, 3] .= 0.;
    testImage[miny:maxy, minx, 1] .= 1.;
    testImage[miny:maxy, minx, 2] .= 0.;
    testImage[miny:maxy, minx, 3] .= 0.;
    testImage[miny:maxy, maxx, 1] .= 1.;
    testImage[miny:maxy, maxx, 2] .= 0.;
    testImage[miny:maxy, maxx, 3] .= 0.;
    return nothing;
end;

########################################################################################
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
