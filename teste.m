# ------------------------------------------------------ #
#           Implementação da Arquitetura LeNet           #
# ------------------------------------------------------ #
#               Feito por: Arthur Gabriel                #
#     DCOMP, Universidade Federal de São João del-Rei    #
#               Email: arthurgabrielbd@gmail.com         #
# ------------------------------------------------------ #

function teste()
    
    pkg load image;
    t = cputime;
    
    # Carregando imagens para treinamento
    [input_train, output_train] = train_db1();
   
    # Carregando imagens para treino
    [input_test, output_test] = test_db1();
    
    printf('Tempo de leitura dos arquivos: %f segundos\n', cputime-t);
    
    # Parâmetros iniciais
    c = 32;
    l = 32;
    pool_size = 2;
    pool_stride = 2;
    type = 'sigmoid';
    epochs = 2400;
    learning_rate = 0.01;
    x = [];
    y_test = [];
    y_train = [];
    filt2 = [];
    error_test = 0;
    error_train = 0;
    
    flayer_1 = randFilters(filter_size=5, filter_amount=10);
    nlayer_1 = randFilters(filter_size=1, filter_amount=10);
    flayer_2 = randFilters(filter_size=5, filter_amount=40);
    
    % Início do treinamento da rede
    [w1, w2] = treinacnn(input_train, output_train, epochs, l, c, pool_size, 
                         pool_stride, flayer_1, nlayer_1, flayer_2, type, 
                         learning_rate);
    printf('Tempo de termino do treino da rede: %f segundos\n', cputime-t);

    # Cálculo do erro do teste
    for i=1:length(input_test)
        saida = (testacnn(input_test{i}, w1, w2, l, c, pool_size, pool_stride, 
                 flayer_1, nlayer_1, flayer_2, type));
        [maxl, argma] = max(saida');
        [maxs, argmaxs] = max(output_test(i, :));
        if argma != argmaxs
            error_test = error_test + 1;
        end
    end
    
    # Cálculo do erro do treinamento    
    for i=1:length(input_train)
        saida = (testacnn(input_train{i}, w1, w2, l, c, pool_size, pool_stride, 
                          flayer_1, nlayer_1, flayer_2, type));
        [maxl, argma] = max(saida');
        [maxs, argmaxs] = max(output_train(i, :));
        if argma != argmaxs
            error_train = error_train + 1;
        end
    end
    
    printf('Erro de acordo com o treino: %f\n', error_train/length(input_train));
    printf('Erro de acordo com o teste: %f\n', error_test/length(input_test));
    #save result.txt x y_train y_test 
    printf('Total cpu time: %f seconds\n', cputime-t);
end

function [w1, w2] = treinacnn(image, y, epochs, l, c, pool_size, pool_stride, 
                              flayer_1, nlayer_1, flayer_2, tipo, lr)
    x=[];
    
    for i=1:length(image)
        [maxp, qf] = featuresmono(image{i}, l, c, pool_size, pool_stride, 
                                  flayer_1, nlayer_1, flayer_2);
        e = [];
        for j=1:qf
            mmxp = max(maxp{j}(:));
            if mmxp == 0
                mmxp = 1;
            end
            e{j} = maxp{j}(:)/mmxp; 
        end
        
        x = [x; cell2mat(e)(:)'];
    end
    
    if max(y) == 0
        maxy = 1;
    else
        maxy = max(y);
    end  
    y = y./maxy;
    
    [output, w1, w2] = treinann(x, y, epochs, tipo, lr);
end

function [maxp, qf] = featuresmono(image, l, c, pool_size, pool_stride, 
                                   flayer_1, nlayer_1, flayer_2)
                                   
    k = 1;
    img = double(image); 
    img = imresize(img, [l, c]); 
    [norm_map, b] = conv_layer(img, flayer_1, nlayer_1, 'relu', 'op');
    pool_1 = maxpool(norm_map, pool_size, pool_stride);
    [a, filters_2] = conv_layer(pool_1, flayer_2, 0,'relu', 'no');
    for j=1:length(filters_2)
        filt2{j} = maxpool(filters_2(:, :, j), pool_size, pool_stride);
        maxp{k} = filt2{j};
        k = k + 1;
    end
    qf = 40;
end

function [kernel] = randFilters(filter_size, filter_amount)
    kernel = ones(filter_size, filter_size, filter_amount);
    
    for i=1:filter_amount
       kernel(:,:,i) = randn(filter_size); 
    end
end

function [norm_map, fe_maps] = conv_layer(img, flayer_1, nlayer_1, activation, normalizer)
    
    [l, c] = size(img);
    fe_maps = ones(l-4, c-4, length(flayer_1(1,1,:)));
    
    for i=1:length(flayer_1)
        fe_maps(:, :, i) = conv2(img, flayer_1(:, :, i), 'valid');
        % ReLU Off
        #if activation == 'relu'
        fe_maps(:, :, i) = maxsoft(fe_maps(:, :, i));
        #end
    end
    
    norm_map = convn(fe_maps, nlayer_1, 'valid');
        % ReLU Off
     #   if activation == 'relu'
    norm_map = maxsoft(norm_map);
     #   end
    if normalizer == 'no'
        norm_map = 0;
    end
end

function [filt] = c1(img, activation)
    mascara = [  0  0  0  0  0;
                 0 -1 -1 -1  0;
                 0 -1  8 -1  0;
                 0 -1 -1 -1  0;
                 0  0  0  0  0]/8;
    filt{1} = conv2(img, mascara, 'valid');
    mascara = [  0  0  0  0  0;
                -1 -2 -3 -2 -1;
                 0  0  0  0  0;
                 1  2  3  2  1;
                 0  0  0  0  0]/9;
    filt{2} = conv2(img, mascara, 'valid');
    mascara = [  0 -1  0  1  0;
                 0 -2  0  2  0;
                 0 -3  0  3  0;
                 0 -2  0  2  0;
                 0 -1  0  1  0]/9;
    filt{3} = conv2(img, mascara, 'valid');
    mascara = [  0 -5  0  1  0;
                 0 -4  0  2  0;
                 0 -3  0  3  0;
                 0 -2  0  4  0;
                 0 -1  0  5  0]/15;
    filt{4} = conv2(img, mascara, 'valid');
    if activation == 'relu'
        for i=1:4
            filt{i} = maxsoft(filt{i});
        end
    end
end

function [filt] = c2(img, activation)
    mascara = [  1  1  1  1  0;
                 1  1  1  0  0;
                 1  1  0  0  0;
                 1  0  0  0  0;
                 0  0  0  0  0]/10;
    filt{1} = conv2(img,mascara, 'valid');
    mascara = [  0  1  1  1  1;
                 0  0  1  1  1;
                 0  0  0  1  1;
                 0  0  0  0  1;
                 0  0  0  0  0]/10;
    filt{2} = conv2(img, mascara, 'valid');
    mascara = [  0  0  0  0  0;
                 1  0  0  0  0;
                 1  1  0  0  0;
                 1  1  1  0  0;
                 1  1  1  1  0]/10;
    filt{3} = conv2(img, mascara, 'valid');
    mascara = [  0  0  1  0  0;
                 0  0  1  0  0;
                 0  0  1  0  0;
                 0  0  1  0  0;
                 0  0  1  0  0]/5;
    filt{4} = conv2(img, mascara, 'valid');
    mascara = [  0  0  0  0  0;
                 0  0  0  0  0;
                 1  1  1  1  1;
                 0  0  0  0  0;
                 0  0  0  0  0]/5;
    filt{5} = conv2(img, mascara, 'valid');
    mascara = [  1  0  0  0  0;
                 0  1  0  0  0;
                 0  0  1  0  0;
                 0  0  0  1  0;
                 0  0  0  0  1]/5;
    filt{6} = conv2(img, mascara, 'valid');
    mascara = [  0  0  0  0  1;
                 0  0  0  1  0;
                 0  0  1  0  0;
                 0  1  0  0  0;
                 1  0  0  0  0]/5;
    filt{7} = conv2(img, mascara, 'valid');
    mascara = [  0  1  1  1  0;
                 1  1  1  1  1;
                 1  1  1  1  1;
                 1  1  1  1  1;
                 0  1  1  1  0]/21;
    filt{8} = conv2(img, mascara, 'valid');
    mascara = [  1  1  1  1  1;
                 1  1  1  1  1;
                 1  1  1  1  1;
                 1  1  1  1  1;
                 1  1  1  1  1]/25;
    filt{9} = conv2(img, mascara, 'valid');
    mascara = [  0  0  0  0  0;
                 0  0  0  0  1;
                 0  0  0  1  1;
                 0  0  1  1  1;
                 0  1  1  1  1]/10;
    filt{10} = conv2(img, mascara, 'valid');
    if activation == 'relu'
        for i=1:10
            filt{i} = maxsoft(filt{i});
        end
    end
end
 
function maxp = pool(filt,l,c)
    maxs = maxsoft(filt);
    maxp = maxpool(maxs,l,c); 
end

function [output, w1, w2] = treinann(x, y, epochs, tipo='sigmoid', lr)
    
    # Inicialização dos pesos com valores aleatórios
    w1 = rand(1000, 1000);
    w2 = rand(7, 1000);
    
    # Camada de entrada da rede
    a{1} = x';

  # Laço que determina quantas iterações serão feitas
    for i=1:epochs 
        if tipo=='sigmoid'
        
            # Cálculo da camada intermediária
            a{2}= 1./ (1+e.^-(w1*a{1}));
            
            # Cálculo da camada de saída
            a{3}= 1./ (1+e.^-(w2*a{2}));
            
            # Backpropagation
            erro_3 = (a{3}-y');
            %erro_2 = (w2' * erro_3).* a{2}.*(1-a{2});
            erro_2 = (w2' * erro_3).* (1./(1+e.^-a{2}));
           
            # Atualização dos pesos da rede
            w1 = w1 - lr*(erro_2 * a{1}');
            w2 = w2 - lr*(erro_3 * a{2}');  
        end 
    end
    
    output = a{3};
end 

function [output] = testann(x, w1, w2)

    # Camada de entrada da rede
    a{1} = x';
 
    # Camada intermediária
    a{2}= 1./ (1+e.^-(w1*a{1}));
        
    # Camada de saída
    a{3}= 1./ (1+e.^-(w2*a{2}));
    
    output = a{3};
end

function [output]=testacnn(img, w1, w2, l, c, p1, p2, flayer_1, nlayer_1, 
                           flayer_2, tipo)
    
    [maxp, qf] = featuresmono(img, l, c, p1, p2, 
                                  flayer_1, nlayer_1, flayer_2);
    
    for i=1:qf
        mmaxp = max(maxp{i}(:));
        if mmaxp == 0
            mmaxp = 1;
        end
        e{i} = maxp{i}(:)/mmaxp;
    end
    x = cell2mat(e);
    
    [output] = testann(x(:)', w1, w2, tipo);
end  


function [maximg]=maxsoft(image)
    [l, c] = size(image);
    for i=1:l
        for j=1:c
            maximg(i, j) = max([0, image(i, j)]);
        end
    end
end

% Faz o pooling e resulta o maior valor
function img = maxpool(img, stride1, stride2)
    img = double(img);
    fun = @(img) max(img(:));
    img = blockproc(img, [stride1 stride2], fun);
end

function img = convs(img, mask, stride1, stride2)
    img = double(img);
    fun = @(img) conv2(img,mask,'valid');
    img = blockproc(img, [stride1 stride2], fun);
end