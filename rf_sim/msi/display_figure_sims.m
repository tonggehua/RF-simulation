% Show excited bins 
clear all
close all
clc 
%% MAVRIC 
load('mavric.mat')
ss = reshape(raw_signal, 29, 48, 5); 
%%
figure(1); 
for y = 1:5
    subplot(5,1,y); hold on 
    imagesc(abs(ss(:,:,y))); axis equal off
    title(sprintf('MAVRIC Bin #%d',y))
end
%%
figure(2); 
bw = 1500; 
load('bmap_msi_fig.mat')
b0 = b0*bw/2;
bdr = linspace(-bw/2,bw/2,6); 
for y = 1:5
    subplot(5,1,y); hold on
    imagesc((bdr(y)<b0&b0<bdr(y+1)))
    title(sprintf('Ideal bin #%d',y))
    axis equal off
end


