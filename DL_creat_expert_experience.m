function [ expert_appentice_vector,map_matrix ] = creat_expert_experience(  )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
   %appentice_matrix=zeros(4,4); 
   appentice_vector=zeros(15,1);
   learn_wall=0;
   gamma=0.9;
   count=0;
   position_x=1;
   position_y=1;
while ~((position_x==20 && position_y==20) || count>50)
   
count=count+1;

direction=input('please give me the expert direction:');
% action = max_index;
map_matrix(position_x,position_y)=count;

pre_position_x=position_x;
pre_position_y=position_y;

switch direction
     
    case 8
        position_x = pre_position_x-1;   %up
    case 2
        position_x = pre_position_x+1;  %down
    case 4
        position_y = pre_position_y-1;  %left
    case 6
        position_y = pre_position_y+1;  %right
    
end

    if(position_x==0 || position_x==21 || position_y==0 || position_y==21)
        learn_wall = learn_wall + power(gamma,count);
        position_x = pre_position_x;
        position_y = pre_position_y;
       %disp(learn_wall);
   
    else
        %appentice_matrix(position_x-1,position_y-1)=appentice_matrix(position_x-1,position_y-1)+power(gamma,count);
        appentice_vector=appentice_vector+encoder(position_x,position_y)*power(gamma,count);
        %disp(appentice_vector);
    end   
    
 
  disp(['position_x: ' num2str(position_x) ' position_y: ' num2str(position_y)]);
 %disp(['count: ' num2str(count)]);   

end
expert_appentice_vector=appentice_vector;
expert_appentice_vector=[expert_appentice_vector;learn_wall];
disp('finish');
end

