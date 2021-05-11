function result = test_svm(svm, test_data, test_label, kerneltype)
% 功能说明：
% 完成测试集的预测以及准确率的输出
% 语法习惯核参数列表：result = test(svm, test_data, test_label, kerneltype)
% input:
% svm: train_svm函数返回的结构体（详见help train_svm）
% test_data: 测试数据
% test_label：测试集标签
% kerneltype：核技巧种类，形式参数，可选：linear gaussian sigmoid mullinear triangle
% output:
% result：结构体，属性如下
% result.Y:测试集中数据的预测类别  result.Y ∈{+1，-1}
% result.accuracy:测试集的准确率

% 教材非线性支持向量机学习算法的策略为选择a的一个正分量0< a <C进行计算
% 此处选择了对所有满足0< ai <C求得bi，并对b进行取平均运算
sum_b = svm.label - (svm.a'.* svm.label)*kernel(svm.data,svm.data,kerneltype);
b = mean(sum_b);
w = (svm.a'.* svm.label)*kernel(svm.data,test_data,kerneltype);% 统一起见，令 w = sigma(ai*yi*K(x,xi)
result.Y = sign(w+b);% 加外壳符号函数进行分类
result.accuracy = size(find(result.Y==test_label))/size(test_label);% 预测正确的数据数目/总测试集数目
end
