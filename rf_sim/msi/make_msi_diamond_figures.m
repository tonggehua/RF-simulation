% Load data
clear all; close all; clc
%% Load data
%load('results_msi2d_test_sigpy180_TEfixed.mat') % TE = 100 ms; 
%load('results_msi2d_test_sigpy180_TEfixed_longTE.mat') % TE = 500 ms;
%e0 = load('results_msi2d_with_spoilers_no-t2star.mat');
load('simulated_Data/results_msi2d_with_spoilers_n25_TE500.mat'); 
%results = e0.results - e1.results; 

%% Functions for boundary lines 
% Oblique
% Find the 4 points
thk = 5; % mm 
rfbw = 500; % Hz 

dfdz = rfbw/thk; 
df_90 = @(z,F) F + dfdz*z;
df_180 = @(z,F) F - dfdz*z;

zmodel = [-16,15]; % mm
f0 = 0;

fmodel_90_upper = df_90(zmodel,f0+rfbw/2);
fmodel_90_lower = df_90(zmodel,f0-rfbw/2);
fmodel_180_upper = df_180(zmodel,f0+rfbw/2);
fmodel_180_lower = df_180(zmodel,f0-rfbw/2); 

% Translate into pixel
zpos = @(zpx) 2.5*zpx - 17.5; 
fpos = @(fpx) (2500/13)*fpx - (1250+2500/13);
zpix = @(z) 0.4*z + 7; 
fpix = @(f) (13/2500)*f + 15/2;
% Draw line 
zp = zpix(zmodel);
fp1 = fpix(fmodel_90_upper);
fp2 = fpix(fmodel_90_lower);
fp3 = fpix(fmodel_180_upper);
fp4 = fpix(fmodel_180_lower);
% Straight
%% 
figure(1); hold on; title('|M_{xy}| at TE')
s = reshape(signals_at_TE, 13, 12); 
imagesc(abs(s)); colorbar; title("|Mxy| at TE")
line(zp,fp1,'Color','red','LineStyle','-','LineWidth',2);
line(zp,fp2,'Color','red','LineStyle','-','LineWidth',2);
line(zp,fp3,'Color','red','LineStyle','-','LineWidth',2);
line(zp,fp4,'Color','red','LineStyle','-','LineWidth',2);
axis equal off 
axis([-0.5,13.5,-0.5,12.5])
%%
figure(2); hold on
Mxyz = {'M_x', 'M_y', 'M_z'};
r = reshape(results, 13, 12, size(results,2), size(results,3));
%r = (r / 10 + 1)/2;

rr = r(:,:,1,:) + 1j*r(:,:,2,:); 
rr = abs(rr); 


r = (r+1)/2;
q = 1; 
p = 1; 
for u = 1:size(results,3)
    for v = 1:4
        subplot(size(results,3),4,q);hold on 
        if v == 1
            imshow(rr(:,:,u)); title('|M_{xy}|')
            c = colorbar('Ticks',0:0.2:1);
        else
            imshow(r(:,:,v-1,u)); title(Mxyz{v-1});
            c = colorbar('Ticks',0:0.2:1, ...
                     'TickLabels',2*((0:0.2:1) - 0.5));
        end
        xlabel('z bin'); ylabel('df bin'); 
        colormap default
        q = q + 1;
    end
end
