% clear
% close all;
% clc;

B=-ones(9,8);
B(4,[1,8])=1;
B(5,1:8)=1;
B(7,5)=1;
figure
imagesc(B)
title('B')

D=-ones(9,8);
D([2,8],[4,5])=1;
D([3,7],[5,6])=1;
D([4,6],[6,7])=1;
D(5,[7,8])=1;
figure
imagesc(D)
title('D')

K=-ones(9,8);
K(1,8)=1;
K(2,7)=1;
K(3,6)=1;
K(4:9,5)=1;
K([8,9],1)=1;
K(9,2:4)=1;
figure
imagesc(K)
title('K')


G=-ones(9,8);
G(1,[5,8])=1;
G(2,[4,7])=1;
G(3,6)=1;
G(4:9,5)=1;
G([8,9],1)=1;
G(9,2:4)=1;
figure
imagesc(G)
title('G')

P=-ones(9,8);
P(4,[1,8])=1;
P(5,1:8)=1;
P(7,[4,5])=1;
P(8,4)=1;
figure
imagesc(P)
title('P')


% hopfild network
input = B; 
net = newhop(input);
Ai = input;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)

input = D;
net = newhop(input);
Ai = input;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)

input = K; 
net = newhop(input);
Ai = input;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)

%data  with 20% noise
 D_noise =1.2*D;
 B_noise =1.2*B;
 K_noise =1.2*K;

input_noise = D_noise;
net = newhop(D);
Ai = input_noise;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)
D_new = reshape(hardlim(Y{1,500}),9,8);

input_noise = B_noise;
net = newhop(B);
Ai = input_noise;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)
B_new = reshape(hardlim(Y{1,500}),9,8);

input_noise = K_noise;
net = newhop(K);
Ai = input_noise;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)
K_new = reshape(hardlim(Y{1,500}),9,8);

figure
imagesc(B_new);
title('B-Saved')

figure
imagesc(K_new);
title('K-Saved')

figure
imagesc(D_new);
title('D-Saved')


input = P; 
net = newhop(input);
Ai = input;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)

 P_noise =1.4*P;

input_noise = P_noise;
net = newhop(P);
Ai = input_noise;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)
P_new = reshape(hardlim(Y{1,500}),9,8);
figure
imagesc(P_new);
title('P-Saved')


input = G; 
net = newhop(input);
Ai = input;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)

 G_noise =1.4*G;

input_noise = G_noise;
net = newhop(G);
Ai = input_noise;
[Y,Pf,Af,E,perf] = sim(net,{8 500},[],Ai)
G_new = reshape(hardlim(Y{1,500}),9,8);
figure
imagesc(G_new);
title('G-Saved')


 
 