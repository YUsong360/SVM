function [train_data,train_label,test_data,test_label,m1,n1,m2,n2] = data_build(data_num,data_features,k)
% [train,test] = data_build(data_num,data_features,k)
% 功能：生成数据并实现训练集和测试集分类
% 语法和参数列表：
% [train_data,train_label,test_data,test_label,m1,n1,m2,n2] = data_build(data_num,data_features,k)
% input：
% data_num -- 数据总量
% data_features -- 特征维数
% k -- 用于训练的比例
% 
% output:
% 返回训练集与测试集的数据及标签（类别）
% 名称一一对应不再注释
% m1,n1,m2,n2分别为train_data，test_data的行列数

data = randn(data_num,data_features);
label = zeros(data_num,1);

for i = 1:data_num
    % 根据数据特征数目设置分类依据
    if data_features == 2
%         if data(i,1)^2+data(i,2)^2 > 1
        if data(i,1)+data(i,2) > 1
            label(i)=1;
        else
            label(i)=-1;
        end
    elseif data_features > 2
        if data(i,1)^2+data(i,2)^2+data(i,3)^2 < 1
            label(i)=1;
        else
            label(i)=-1;
        end
    end
end

    
% 乱序排列
randIndex = randperm(data_num);
data_new=data(randIndex,:);
label_new=label(randIndex,:);

k = k*data_num;
train_data=data_new(1:k,:);
train_label=label_new(1:k,:);
test_data=data_new(k+1:end,:);
test_label=label_new(k+1:end,:);
[m1,n1] = size(train_data);
[m2,n2] = size(test_data);
end

