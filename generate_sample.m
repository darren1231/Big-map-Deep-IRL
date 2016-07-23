function [ test_x,test_y ] = generate_sample(number)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    
    number=10000;
    for i=1:number
    rand_x = floor(mod(rand*100,20))+1;
    rand_y = floor(mod(rand*100,20))+1;
  
    matrix = produce_state_picture(rand_x,rand_y);
    
    test_x(i,:)=matrix;
    index=20*(rand_x-1)+rand_y;
    zero_y= zeros(1,400);
    zero_y(index)=1;
    test_y(i,:)=zero_y;  
        if mod(i,1000)==0
        disp(i);
        end
    end
    save('test_x_20(60_black).mat','test_x');
    save('test_y_20(60_black).mat','test_y');
    disp('Done');
end

