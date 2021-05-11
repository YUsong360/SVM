%% 数据预处理和导入
close;clear;clc
[train_data,train_label,test_data,test_label,m1,n1,m2,n2] = data_set(0.6,100);
%% 模型训练
Kernel = 'linear';% Kernel 核技巧备选：gaussian linear sigmoid mullinear 
svm = train_svm(train_data',train_label',Kernel,10); % svm = train_svm(X,Y,kertype,C) C为变量上界（惩罚因子） svm为结构体
%% 模型测试
result = test_svm(svm,test_data',test_label',Kernel);
fprintf('训练完成！\n应用模型：SVM 支持向量机\n优化算法:interior-point-convex\n核函数：%s\n测试集识别率为：%f\n',Kernel,result.accuracy);
%% 作图显示数据以及训练结果；中间为支持向量[三维]
draw_svm(train_data,train_label,svm,3,Kernel);