function [train_data,train_label,test_data,test_label,m1,n1,m2,n2] = data_set(k,data_num)
% 功能说明：完成数据的预处理，setosa：1  versicolor：2  virginica：3
% 调用语法及参数说明：[data_iris,data_label] = data_set();
% 
load('data_iris.mat');load('data_label.mat');
data_label = zeros(data_num,1);
for i = 1:data_num
    switch species(i)
        case 'setosa'
            data_label(i) = 1;
        case 'versicolor'
            data_label(i) = -1;
%         case 'virginica'
%             data_label(i) = 3;
    end
end
data_iris = iris(1:data_num,:);

% 乱序排列
randIndex = randperm(data_num);
data_new=data_iris(randIndex,:);
label_new=data_label(randIndex,:);

% 分为两组，比例k用于训练，剩余用于测试
k = k*data_num;
train_data=data_new(1:k,:);
train_label=label_new(1:k,:);
test_data=data_new(k+1:end,:);
test_label=label_new(k+1:end,:);
[m1,n1] = size(train_data);
[m2,n2] = size(test_data);
end

