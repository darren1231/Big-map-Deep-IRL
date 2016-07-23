clear;
load('train_x_20(60_black).mat');
load('train_y_20(60_black).mat');

load('test_y_20(60_black).mat');
load('test_x_20(60_black).mat');


% train_x = double(train_x)/255;
% test_x  = double(test_x)/255;
% train_y = double(train_y);
% test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0);
sae = saesetup([3600 1800 900 800 500]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.5;

sae.ae{4}.activation_function       = 'sigm';
sae.ae{4}.learningRate              = 1;
sae.ae{4}.inputZeroMaskedFraction   = 0.5;

% sae.ae{5}.activation_function       = 'sigm';
% sae.ae{5}.learningRate              = 1;
% sae.ae{5}.inputZeroMaskedFraction   = 0.5;
% 
% sae.ae{6}.activation_function       = 'sigm';
% sae.ae{6}.learningRate              = 1;
% sae.ae{6}.inputZeroMaskedFraction   = 0.5;
% 
% sae.ae{7}.activation_function       = 'sigm';
% sae.ae{7}.learningRate              = 1;
% sae.ae{7}.inputZeroMaskedFraction   = 0.5;

opts.numepochs =   30;
opts.batchsize = 1000;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([3600 1800 900 800 500 400]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
% nn.W{1} = sae.ae{1}.W{1};
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% nn.W{5} = sae.ae{5}.W{1};
% nn.W{6} = sae.ae{6}.W{1};
% nn.W{7} = sae.ae{7}.W{1};

w1 = sae.ae{1}.W{1};
w2 = sae.ae{2}.W{1};
w3 = sae.ae{3}.W{1};
w4 = sae.ae{4}.W{1};
% w5 = sae.ae{5}.W{1};
% w6 = sae.ae{6}.W{1};
% w7 = sae.ae{7}.W{1};

%save('autoencoder_w_1600 800 400 200 100.mat','w1','w2','w3','w4','w5','w6','w7');
save('autoencoder_w_3600 1800 900 800 500.mat','w1','w2','w3','w4');

% Train the FFNN
opts.numepochs =   30;
opts.batchsize = 1000;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
disp(er);
%assert(er < 0.16, 'Too big error');
