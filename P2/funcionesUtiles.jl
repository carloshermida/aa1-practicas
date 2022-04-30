push!(LOAD_PATH, ".")
module funcionesUtiles
    using JLD2
    using Images

    # Functions that allow the conversion from images to Float64 arrays
    imageToGrayArray(image:: Array{<:Colorant,2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
    function imageToColorArray(image::Array{<:Colorant,2})
        matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
        image = RGB.(image);
        matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
        matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
        matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
        return matrix;
    end;

    # Some functions to display an image stored as Float64 matrix
    # Overload the existing display function, either for graysacale or color images
    import Base.display
    display(image::Array{Float64,2}) = display(Gray.(image));
    display(image::Array{Float64,3}) = (@assert(size(image,3)==3); display(RGB.(image[:,:,1], image[:,:,2], image[:,:,3])); )

    # Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
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

    # Functions to load the dataset
    function loadTrainingDataset()
        (positivesColor, positivesGray) = loadFolderImages("positivos");
        (negativesColor, negativesGray) = loadFolderImages("negativos");
        targets = [trues(length(positivesColor)); falses(length(negativesColor))];
        return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
    end;
    loadTestDataset() = ((colorMatrix,_) = loadFolderImages("testFinal"); return colorMatrix; );

    # Function to set a red box on a window on a color image represented as a 3D Float64 Array
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




    # # Load the dataset
    # (colorDataset, grayDataset, targets) = loadTrainingDataset();
    # # Display an example of image
    # display(colorDataset[15]);
    # # Load the final test images
    # colorTestDataset = loadTestDataset();
    # # Put a red box on these coordinates
    # setRedBox!(colorTestDataset[1], 10, 100, 10, 200);
    # # Display this image
    # display(colorTestDataset[1]);


    # Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
    function loadFolderImagesTest(folderName::String)
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
        return images;
    end;
end;
