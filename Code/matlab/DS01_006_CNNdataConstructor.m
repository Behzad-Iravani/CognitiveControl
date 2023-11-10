% Post processing of CNN
% Author: Behzad Iravani and Neda Kaboodvand
% behzadiravani@gmail.com; n.kaboodvand@gmail.com
%% Start 
%%------ Clear memory -----%%
clc
clear 
% ---  Create CNN data object ----%% 
filepath = 'C:\Users\behira\OneDrive - Karolinska Institutet\Desktop\Manuscript\CognitiveControl\scripts\data\interp\python';
disp('creating CNNdata object')
Data    = CNNdata(filepath,{'history.csv', 'accuracy.txt',...
                    'activation.mat', 'MeanAverageError.mat','grads_data_time.mat'...
                    'MeanAverageError_chan.mat', 'Kernel_layer.mat',...
                    'Maximized_class.mat', 'nodeLabel.xlsx', 'data_incong_hit_err.mat', 'data_partitioned_prepratory.mat'});
Data.LoadParameters()
disp('saving CNNdata')
save('C:\Users\behira\OneDrive - Karolinska Institutet\Desktop\Manuscript\CognitiveControl\scripts\data\interp\matlab\CNNData.mat', 'Data', '-v7.3')
disp('Done! $END')
% $End