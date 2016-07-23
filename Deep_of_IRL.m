%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------------------------------------------
% Project <A simple way to implement Deep IRL>
% Date    : 2016/6/23
% Author  : Kun da Lin
% Comments: Language: Matlab. 
% Source: matlab 
% ---------------------------------------------------------------------------------
%Q(state,x1)=  oldQ + alpha * (R(state,x1)+ (gamma * MaxQ(x1)) - oldQ);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
%% set paramater
round_max=20000;
%N=4;
stop_count=100;

%%  zero initial
qtable = zeros(20,20,4);
round = 0;
count=0;


%% You can use next line create your own experience and save it for using in the future
%[ expert_vector ,expert_map_matrix]= creat_expert_experience(N);
%[ expert_vector ,expert_map_matrix]= DL_creat_expert_experience;
%save('DL_expert.mat','expert_vector','expert_map_matrix');
load('DL_expert.mat');


%% Randomly create experience of learner
[ learn_appentice_vector ] = DL_creat_learner_experience(stop_count);

%% IRL initialization , create firt set of omega(reward matrix)
w=expert_vector-learn_appentice_vector;
[ reward_matrix ] = get_reward_matrix( w );
wall_reward=w(end);
last_reward_matrix=reward_matrix;

%% Initial F,a    
F=ones(16,1)*0.9;% add one is for the wall ,15 is neuron
IRL_a=0.1;
IRL_terminate=0;% if IRL_terminate=1 mean :find 

%% Learning start:
while round<round_max
    round=round+1;

map_matrix=zeros(20,20);


%% Caculate the square error
delta_mu=expert_vector-learn_appentice_vector;
square_error=sqrt(sum(power(delta_mu,2)));
save_error(round+1,1)=sum(square_error);
disp(['square_error: ' num2str(square_error)]);


if(square_error<0.000001)
     IRL_terminate=1;
     %save('reward_expert_15.mat','reward_matrix','wall_reward');
     disp(['Congratulation! You succeed at ' num2str(round) ' trial']);
    % break;
else
    IRL_terminate=0;
     %qtable = zeros(N+2,N+2,4);   %don't reset qtable is faster than reset
end

 %%  IRL:caculate new reward function
 % If round ==0 or  IRL_terminate==1(mean you success) break if
if(~(round==1 || IRL_terminate==1))
for i=1:16
    if(expert_vector(i,1)==learn_appentice_vector(i,1))                                                                                                                                 
        F=F*(1-IRL_a);
    else
        F=F+(1-F)*IRL_a;
    end
end
w= w+F.*delta_mu;

[ reward_matrix ] = get_reward_matrix( w );
% disp(reward_matrix-last_reward_matrix);
% last_reward_matrix=reward_matrix;
reward_matrix=normc(reward_matrix);
mesh(reward_matrix);


wall_reward=w(end);   
end

%% RL : Once you got the reward function , throw into RL and train it.
RL_episode=0;
frame=0;
 while (RL_episode<10)
     RL_episode=RL_episode+1;

%% Initial some parameters
    RL_count=0;
    position_x=1;
    position_y=1;
while ~((position_x==20 && position_y==20) || RL_count>stop_count) %stop_count
    RL_beta=0.7;
	RL_gamma=0.9;


RL_count=RL_count+1;
rand_action = floor(mod(rand*10,4))+1;
rand_number=rand;
[max_q, max_index] = max([qtable(position_x,position_y,1) qtable(position_x,position_y,2) qtable(position_x,position_y,3) qtable(position_x,position_y,4)]);

if(qtable(position_x,position_y,rand_action)>=qtable(position_x,position_y,max_index))
    action = rand_action;
else
    action = max_index;
end

%% show picture    
    frame=frame+1;
    matrix=produce_state_picture( position_x,position_y );
    I=reshape(matrix,40,40);
    imshow(I);
    imshow(I,[]); 
    imshow(I,'InitialMagnification','fit');
    Map(frame)=getframe;

%% epsilon greedy
%epsilon=1-1/(1+exp(-round));
%epsilon=1-round/50;
if(rand_number<0.05)
    action = rand_action;
else
    action = max_index;
end
map_matrix(position_x,position_y)=RL_count;

pre_position_x=position_x;
pre_position_y=position_y;

switch action
     
    case 1
        position_y = pre_position_y-1;   %up
        position_x = pre_position_x;
    case 2
        position_y = pre_position_y+1;  %down
        position_x = pre_position_x;
    case 3
        position_x = pre_position_x-1;  %left
        position_y = pre_position_y;
    case 4
        position_x = pre_position_x+1;  %right
        position_y = pre_position_y;
end
    %wall
    if(position_x==0 || position_x==21 || position_y==0 || position_y==21)  
        position_x = pre_position_x;
        position_y = pre_position_y;
        reward=wall_reward;
        b=0;   
        
    %goal
    elseif(position_x==20 && position_y==20)
             %% show picture         
        matrix=produce_state_picture( position_x,position_y );
        I=reshape(matrix,40,40);
        imshow(I);
        imshow(I,[]); 
        imshow(I,'InitialMagnification','fit');
        Map(frame)=getframe;
        
        reward=reward_matrix(position_x,position_y);
        b=0;
    else        
        reward=reward_matrix(position_x,position_y);
    end  
    
    
  [max_qtable, max_qtable_index] = max([qtable(position_x,position_y,1) qtable(position_x,position_y,2) qtable(position_x,position_y,3) qtable(position_x,position_y,4)]);
  %disp(['position_x: ' num2str(position_x) ' position_y: ' num2str(position_y)]);
  %disp(['count: ' num2str(count)]);
 
   %% Q learning
   old_q=qtable(pre_position_x,pre_position_y,action);
   new_q=old_q+RL_beta*(reward+RL_gamma*max_qtable-old_q);
   qtable(pre_position_x,pre_position_y,action)=new_q;   
   
   %disp(['RL_Epsidoe: ' num2str(RL_episode) ' RL_count: ' num2str(RL_count)]);
end   
   %disp(['RL_Epsidoe: ' num2str(RL_episode) ' trial: ' num2str(round)]);
   
end %episode
%% Once you have done the work of RL, use the Q table you obtain and throw it into next function to find mul
 [ learn_appentice_vector,learner_map_matrix] = DL_transport_qtable_to_mul(stop_count,qtable);
  disp(['trial: ' num2str(round) ' count: ' num2str(RL_count)]);
  save_data(round,:)=RL_count;  
end

 figure(1);
 plot(save_data);
 title('Learning');
 xlabel('episode'); % x-axis label
 ylabel('step'); % y-axis label
  
 figure(2)
 plot(save_error);
 title('Error');
 xlabel('episode'); % x-axis label
 ylabel('U_e-U_l'); % y-axis label
 
 figure(3);
 reward_matrix=normc(reward_matrix);
 mesh(reward_matrix);
 title('Reward function');
 xlabel('X'); % x-axis label
 ylabel('Y'); % y-axis label
 zlabel('reward');
 
 disp('expert:');
 disp(expert_map_matrix);
 disp('learner:');
 disp(learner_map_matrix);
 
 