function draw_svm(train_data,train_label,svm,data_features,Kernel)
% 功能说明：根据数据特征的维数判断，进而分别绘图
% 函数语法及参数列表：draw_svm(inputArg1,data_features)
% input: 
% train_data: 训练数据集 
% train_label：训练集数据的类别
% svm：svm结构体(详见train_svm，help train_svm)
% data_features特征维数

switch data_features
    case 2
        plot(train_data(train_label==1,1),train_data(train_label==1,2),'ro',train_data(train_label==-1,1),train_data(train_label==-1,2),'go');hold on;
        plot(svm.data(1,:),svm.data(2,:),'mo');hold on;title(['样本分布',Kernel]); % 显示支持向量 'mo'品红色的圈
    otherwise
        plot3(train_data(train_label==1,1),train_data(train_label==1,2),train_data(train_label==1,3),'r.');hold on;
        plot3(train_data(train_label==-1,1),train_data(train_label==-1,2),train_data(train_label==-1,3),'gx');hold on;
        plot3(svm.data(1,:),svm.data(2,:),svm.data(3,:),'mo');hold on;
        title(['样本分布',Kernel]);
end
end

