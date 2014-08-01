function patches = sampleIMAGES(image)
% sampleIMAGES
% Returns 10000 patches for training

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

[W, H, n_pictures] = size(image);

% D1,D2中的点作为起始位置
D1 = randi(W-patchsize,numpatches,1);
D2 = randi(H-patchsize,numpatches,1);
D3 = randi(n_pictures, numpatches, 1);

for i = 1:numpatches
    patch = image(D1(i)+(1:patchsize),D2(i)+(1:patchsize),D3(i));
    patches(:,i) = patch(:); % column-wise
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches); % range: [0.1, 0.9]

end

