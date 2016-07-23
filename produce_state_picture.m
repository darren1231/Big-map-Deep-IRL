function [ matrix ] = produce_state_picture( x,y )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
I=imread('60_black.bmp');
%I=I*0;
    for x_pix=3*(x-1)+1:3*x
        for y_pix=3*(y-1)+1:3*y
            I(x_pix,y_pix)=255;
        end
    end
    N=imnoise(I,'gaussian');
    imshow(N);
    matrix=reshape(N,1,3600);
    matrix=double(matrix)/255;
end

