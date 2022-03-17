% Load 
clear; clc
load('t2star_hist_data.mat')
%% Generate predicted values
TE = 200e-3; 
t2 = 200e-3;
t2star = 100e-3; 
names = {'SE (no T2*)','SE (T2*)','GRE (no T2*)','GRE (T2*)'};

pv = [exp(-TE/t2),exp(-TE/t2),exp(-TE/t2),exp(-TE/t2star)];

%%
figure(1); hold on  
q = 1;
for u = 1:3
    for v = 1:4
        subplot(3,4,q); hold on;
        title(names{v})
        histogram(squeeze(results(u,:,v)),'BinWidth',0.05)
        axis([0,1,0,50])
        line([pv(v),pv(v)],[0,50],'Color','red','LineWidth',1)
        m = mean(results(u,:,v)); 
        line([m,m],[0,50],'Color','blue','LineWidth',1)
        if q == 12
            legend('Simulated values','True signal','Avg. simulated signal')
        end
        q = q + 1; 
        xlabel('Signal (a.u.)'); ylabel('Count')
    end
end