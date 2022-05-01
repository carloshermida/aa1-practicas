
#                  PRÁCTICA 2 APRENDIZAJE AUTOMÁTICO I
#         Nina López | Borja Souto | Carmen Lozano | Carlos Hermida


push!(LOAD_PATH, ".")
# Importamos los mmódulos donde tenemos las otras funciones que nos podrían hacer
# falta
using Main.modelComparison
using Main.funcionesUtiles
using Images
using JLD2
using Statistics
import Base.display
import FileIO


############################## FUNCIONES ##############################

##### symmetryVH
# Con esta función miramos la simetría de las imágenes. Lo que hacemos es crear
# dos nuevas imágenes en forma de array, una de ellas igual a la original pero
# girada según el eje vertical y la otra lo mismo pero según el eje horizontal.
# A continuación, obtenemos el mismo píxel en una imagen que en la otra y restamos
# los valores, de manera que si son simétricos (valores parecidos) esta diferencia
# va a ser pequeña, mientras que si son muy distintos, resultará un valr muy grande.
# Finalmente, de estas dierencias, obtenemos la media y desviación típica y las
# devlvemos en forma de matriz fila
function symmetryVH(window::Array{Float64,2})
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

    result = zeros(1,4)
    result[1,1] = mean(VError)
    result[1,2] = std(VError)
    result[1,3] = mean(HError)
    result[1,4] = std(HError)

    return result
end

####symmetryH
# Hacemos lo mismo que en la función anterior, pero en este caso tenemos solo en
# cuenta la simetría respecto al eje horizontal
function symmetryH(window::Array{Float64,2})
    rows=size(window,1)
    columns=size(window,2)
    HorizontalSymmetrical=zeros(rows, columns)
    for i=1:rows
        for j=1:columns
            HorizontalSymmetrical[i,j]=window[rows+1-i, j]
        end
    end
    HError=abs.(window-HorizontalSymmetrical)

    result = zeros(1,2)
    result[1,1] = mean(HError)
    result[1,2] = std(HError)

    return result
end

##### differences
# Con esta función miramos la diferencia entre un pixel y el de al lado para
# obtener "bordes". Esto lo hacemos por filas y guardamos esto en una matriz
# Con esta matriz, lo que hacemos es coger la media de color de las tres filas
# superiores, las tres inferiores y las tres del medio, ya que en clases ventana
# con un ojo, estos valores más o menos coinciden en todas.
function differences(grayImage)
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

    # OPCIÓN 1
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

# (TEST DIFFERENCES)
# Comprobamos con dos ventanas (una positiva y otra negativa) el funcionamiento
# de la función anterior
img_path = "positivos/image_0001-1.bmp";
img = load(img_path);
display(img)
grayImage = funcionesUtiles.imageToGrayArray(img);
grayImage = grayImage/mean(grayImage) #BRILLO
x = differences(grayImage)
#display(x)


img_path = "negativos/2.bmp";
img = load(img_path);
display(img)
grayImage = funcionesUtiles.imageToGrayArray(img);
grayImage = grayImage/mean(grayImage) #BRILLO
x = differences(grayImage)
#display(x)


##### sixDivision
# Con esta función lo que haremos es dividir el ancho de una ventana entre 6
# y crear subventanas para mirar sus caracteríticas por separado
function sixDivision(window)
    columns = size(window)[2]
    x = div(columns,6)
    res = columns % 6
    divisions = ones(6)*x
    for i = 1:res
        divisions[i] += 1
    end

    for i = 2:length(divisions)
        divisions[i] += divisions[i-1]
    end

    divisions = Int.(divisions)
    windows = convert(Array{Any,1}, zeros(0))
    push!(windows, window[:,(1:divisions[1])])

    for j = 1:5
        push!(windows, window[:,(divisions[j]+1:divisions[j+1])])
    end

    return windows

end



##### featureExtraction1
# Esta función se corresponde al caso de la aproximación 1, en la que tan solo
# empleamos la media y desviación típica de los colores y del blanco
function featureExtraction1(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i]) #Brillo
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        char = zeros(1, 2)
        char[1,1] = mean(window)
        char[1,2] = std(window)
    end

    return char
end

##### featureExtraction2
# Esta función se corresponde al caso de la aproximación 2, en la que
# empleamos la media y desviación típica de los colores y del blanco, así como la
# simetría de las capas de las matrices de colores y la de blanco y negro
function featureExtraction2(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
            char = hcat(char, symmetryVH(color))
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        char = zeros(1, 2)
        char[1,1] = mean(window)
        char[1,2] = std(window)
        char = hcat(char, symmetryVH(window))
    end

    return char
end

##### featureExtraction3
# Esta función se corresponde al caso de la aproximación 3, en la que
# empleamos la media y desviación típica de los colores y del blanco, así como la
# simetría respecto al eje horizontal de las capas de las matrices de colores
function featureExtraction3(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
            char = hcat(char, symmetryH(color))
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        char = zeros(1, 2)
        char[1,1] = mean(window)
        char[1,2] = std(window)
    end

    return char
end


##### featureExtraction4
# Esta función se corresponde al caso de la aproximación 4, en la que
# empleamos la media y desviación típica de los colores, así como la
# simetría respecto a los dos ejes de las capas de las matrices de colores y los
# "bordes" de las imágenes en blanco y negro
function featureExtraction4(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
            char = hcat(char, symmetryH(color))
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        char = differences(window)
    end

    return char
end


##### featureExtraction5
# Esta función se corresponde al caso de la aproximación 5, en la que
# empleamos la media y desviación típica de los colores, así como la
# media y desviación típica de la cantidad de blanco de las subventanas
function featureExtraction5(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        parts = sixDivision(window)
        char = zeros(0)
        for p in parts
            push!(char, mean(p))
            push!(char, std(p))
        end
        char = char'
    end

    return char
end

##### featureExtraction6
# Esta función se corresponde al caso de la aproximación 6, en la que
# empleamos la media y desviación típica de los colores y del blanco, así como la
# simetría  de las capas de las matrices de colores
function featureExtraction6(window::Array{Float64})
    if typeof(window)==Array{Float64, 3}
        char = zeros(1,6)
        cnt = 1
        for i=1:3
            color = window[:,:,i]/mean(window[:,:,i])
            char[1,cnt] = mean(color)
            cnt += 1
            char[1,cnt] = std(color)
            cnt += 1
            char = hcat(char, symmetryVH(color))
            # char = hcat(char, symmetryH(color))
        end

    elseif typeof(window)==Array{Float64, 2}
        window = window/mean(window)
        parts = sixDivision(window)
        char = zeros(1,4)
        char[1,1] = mean(parts[2])
        char[1,2] = std(parts[2])
        char[1,3] = mean(parts[5])
        char[1,4] = std(parts[5])
    end

    return char
end


##### ClassifyEye
# Esta función recibe una ventana y decide si en ella hay un ojo o no a partir
# de las caraterísticas que se obtienen con las featureExtraction (dependiendo
# de la aprpximación indicada escoge una u otra)
# Con la probabilidad que obtiene la red, y en función del umbral (threshold)
# indicado, decide si se considera como un ojo o no
function ClassifyEye(img, rna, threshold, NormalizationParameters, aprox)
    ColorDataset = funcionesUtiles.imageToColorArray(img)
    GrayDataset = funcionesUtiles.imageToGrayArray(img)
    if aprox == 1
        charC = featureExtraction1(ColorDataset)
        charG = featureExtraction1(GrayDataset)
    elseif aprox == 2
        charC = featureExtraction2(ColorDataset)
        charG = featureExtraction2(GrayDataset)
    elseif aprox == 3
        charC = featureExtraction3(ColorDataset)
        charG = featureExtraction3(GrayDataset)
    elseif aprox == 4
        charC = featureExtraction4(ColorDataset)
        charG = featureExtraction4(GrayDataset)
    elseif aprox == 5
        charC = featureExtraction5(ColorDataset)
        charG = featureExtraction5(GrayDataset)
    elseif aprox == 6
        charC = featureExtraction6(ColorDataset)
        charG = featureExtraction6(GrayDataset)
    end
    char = hcat(charC, charG)
    normalizeMinMax!(char, NormalizationParameters)
    result = rna(char')
    if result[1] >= threshold
        return 1
    else
        return 0
    end
end

############################### CÓDIGO ###############################
# Obtenemos los datos de cada imagen
(colorDataset, grayDataset, targets) = funcionesUtiles.loadTrainingDataset();
#Calculamos el tamaño de ventanas que va a usar en el test
minWindowSizeY = minimum(size.(colorDataset, 1));
maxWindowSizeY = maximum(size.(colorDataset, 1));
minWindowSizeX = minimum(size.(colorDataset, 2));
maxWindowSizeX = maximum(size.(colorDataset, 2));
# Obtenemos todas las imágnes de test
images = funcionesUtiles.loadFolderImagesTest("testFinal")

######################################################################
########################### APROXIMACIÓN 1 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction1.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction1.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Comenzamos probando el funcionamiento con validación cruzada
# Definimos los parámteros
k = 10;
topology = [4];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;
# Creamos el diccionario que los guarda y que le pasaremos a la red
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputs, targets, k)

# Vistos los resultados anteriores, probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([4], trainingDataset);
detectionThreshold = 0.8;
# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 1) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end

######################################################################
########################### APROXIMACIÓN 2 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction2.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction2.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Comenzamos probando el funcionamiento con validación cruzada
# Definimos los parámteros
k = 10;
topology = [2, 2];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;
# Creamos el diccionario que los guarda y que le pasaremos a la red
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputs, targets, k)

# Vistos los resultados anteriores, probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([2, 2], trainingDataset);
detectionThreshold = 0.85;

# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 2) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end


######################################################################
########################### APROXIMACIÓN 3 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction3.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction3.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([2, 2], trainingDataset);
detectionThreshold = 0.85;

# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 3) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end


######################################################################
########################### APROXIMACIÓN 4 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction4.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction4.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Comenzamos probando el funcionamiento con validación cruzada
# Definimos los parámteros
k = 10;
topology = [2, 2];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;
# Creamos el diccionario que los guarda y que le pasaremos a la red
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputs, targets, k)

# Probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([2, 2], trainingDataset);
detectionThreshold = 0.85;

# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 4) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end

######################################################################
########################### APROXIMACIÓN 5 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction5.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction5.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Comenzamos probando el funcionamiento con validación cruzada
# Definimos los parámteros
k = 10;
topology = [2, 2];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;
# Creamos el diccionario que los guarda y que le pasaremos a la red
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputs, targets, k)

# Probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([2, 2], trainingDataset);
detectionThreshold = 0.85;

# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 5) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end


######################################################################
########################### APROXIMACIÓN 6 ###########################
######################################################################
# Cargamos los dataset y extraemos las características
inputsC = vcat(featureExtraction6.(colorDataset)...); #Caracteristicas de colores
inputsG = vcat(featureExtraction6.(grayDataset)...); #Caracterísitcas de blanco
inputs = [inputsC inputsG]; # Las juntamos
positive = colorDataset[targets .== 1]; ##########################
# Normalizamos los inputs
normalizeMinMax!(inputs);
#Obtenemos los parámetros de normalización para usarlos en el test
NormalizationParameters = calculateMinMaxNormalizationParameters(inputs);
# Creamos el dataset de entrenamiento
targetsMatrix = reshape(targets, length(targets), 1);
trainingDataset = (inputs, targetsMatrix)

# Comenzamos probando el funcionamiento con validación cruzada
# Definimos los parámteros
k = 10;
topology = [2, 2];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;
# Creamos el diccionario que los guarda y que le pasaremos a la red
parameters = Dict();
parameters["topology"] = topology;
parameters["learningRate"] = learningRate;
parameters["validationRatio"] = validationRatio;
parameters["numExecutions"] = numRepetitionsAANTraining;
parameters["maxEpochs"] = numMaxEpochs;
parameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, parameters, inputs, targets, k)

# Probamos ahora con el test real
# Para ello primero entrenamos la red e indicamos el valor a partir del que
# consideraremos que son ojos
(rna,losses) = trainClassANN([2, 2], trainingDataset);
detectionThreshold = 0.85;

# Hacemos el test final para cada imagen
for img in images
    windowLocations = Array{Int64,1}[];
    for windowWidth = minWindowSizeX:4:maxWindowSizeX
        for windowHeight = minWindowSizeY:4:maxWindowSizeY
            for x1 = 1:10:size(img,2)-windowWidth
                for y1 = 1:10:size(img,1)-windowHeight
                    x2 = x1 + windowWidth;
                    y2 = y1 + windowHeight;
                    if ClassifyEye(img[y1:y2, x1:x2], rna, detectionThreshold, NormalizationParameters, 6) == 1
                        push!(windowLocations, [x1, x2, y1, y2]);
                    end
                end
            end
        end
    end

    testImage = funcionesUtiles.imageToColorArray(img);
    for i=1:size(windowLocations)[1]
        funcionesUtiles.setRedBox!(testImage, windowLocations[i][1],windowLocations[i][2],windowLocations[i][3],windowLocations[i][4])
    end
    display(testImage)
end
