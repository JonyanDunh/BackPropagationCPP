
//《深度学习中的误差反向传播法基于c++的实现方法 BY JONYANDUNH》
// 本项目以Mnist手写集为数据，共10000条数据，样本位于目录下的t10k-images.idx3-ubyte与t10k-labels.idx1-ubyte中
// 项目地址：https://github.com/JonyanDunh/BP_OF_DL_CPP
// 项目创建日期：2022.10.11
//bilibili：JonyanDunh https://space.bilibili.com/96876893 （作者也是bilibili动态头像的发明者）
//github:JONYANDUNH https://github.com/JonyanDunh
//QQ：1309634881
//Wechat:jonyandunh
#include <iostream>
#include <Eigen/Dense>
/*
引入了矩阵库
Eigen在windows下的安装大家可以参考：http://blog.csdn.net/abcjennifer/article/details/7781936
Linux下的安装可以参考：https://zhuanlan.zhihu.com/p/36706885
Eigen常用函数：https://www.cnblogs.com/ybqjymy/p/16370560.html
*/
#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <string>
#include <vector>
using namespace Eigen;
using namespace std;
struct grads {//定义了一个用于存储梯度的struct
	MatrixXd W1;
	MatrixXd b1;
	MatrixXd W2;
	MatrixXd b2;
};
class SigmoidLayer//定义了Sigmoid层
{
public:
	MatrixXd out;//用于存储网络正向运行时的输出
	MatrixXd forward(MatrixXd x) {
		MatrixXd re_x = -1 * x;
		out = 1 / (re_x.array().exp() + 1);
		return out;
	};
	MatrixXd backward(MatrixXd dout) {
		return dout.array() * (1.0 - out.array()) * out.array();

	};
};
class MulLayer {//定义了乘法层
public:
	MatrixXd x, y;//用于存储网络正向运行时输入的x,y
	MatrixXd forward(MatrixXd forward_x, MatrixXd forward_y) {
		x = forward_x;
		y = forward_y;
		return forward_x * forward_y;
	};
	MatrixXd* backward(MatrixXd dout) {
		static  MatrixXd result[] = { dout * y,dout * x };
		return result;
	};
};
class AddLayer//定义了加法层
{
public:
	MatrixXd x, y;//用于存储网络正向运行时输入的x,y
	MatrixXd forward(MatrixXd forward_x, MatrixXd forward_y) {
		x = forward_x;
		y = forward_y;
		return forward_x + forward_y;
	};
	MatrixXd* backward(MatrixXd dout) {
		static  MatrixXd result[] = { dout,dout };
		return result;
	};
};
class ReLULayer {//定义了ReLU层
public:
	Matrix<bool, Dynamic, Dynamic> mask;//用于存储网络正向运行时输入x矩阵里小于或等于0元素的坐标
	MatrixXd forward(MatrixXd x) {
		mask = (x.array() <= 0);
		for (int i = 0; i < x.rows(); i++)
		{

			for (int z = 0; z < x.cols(); z++)
			{
				if (mask(i, z) == true)//当x大于0时，输出原始值，反之为0
					x(i, z) = 0;
			}

		}
		return x;
	};
	MatrixXd backward(MatrixXd dout) {
		for (int i = 0; i < dout.rows(); i++)
		{
			for (int z = 0; z < dout.cols(); z++)
			{
				if (mask(i, z) == true)//当正向运行时的x大于0，则导数为1，反之导数为0
					dout(i, z) = 0;
			}
		}
		return dout;
	};
};
class SoftmaxLayer {//定义了Softmax层
public:
	MatrixXd forward(MatrixXd x) {
		for (int i = 0; i < x.rows(); i++)
		{
			x.row(i) = x.row(i).array().exp() / x.row(i).array().exp().sum();//算出每个元素的概率
		}
		return x;
	};
};
class Cross_entropy_error_Layer {//定义了 Cross_entropy_error层（用于计算误差）
public:
	double forward(MatrixXd y, MatrixXd t) {
		double delta = 1e-7;
		double sum = 0;
		for (int i = 0; i < y.rows(); i++)
		{
			for (int z = 0; z < y.cols(); z++)
			{
				if (t(i, z) == 1)
				{
					sum += log(y(i, z) + delta);
				}
			}
		}
		return  -sum / y.rows();//返回误差值
	};
};
class AffineLayer {//定义了Affine层

public:
	MatrixXd W;//存储正向传播时的权重
	MatrixXd b;//存储正向传播时的偏置
	MatrixXd x;//存储正向传播时的遍历
	MatrixXd dW;//存储反向传播求出的权重的导数
	MatrixXd db;//存储反向传播求出的偏置的导数
	void init(MatrixXd W1, MatrixXd b1) {

		W = W1;
		b = b1;
	}
	MatrixXd forward(MatrixXd x1) {
		x = x1;
		MatrixXd result = x * W;//相乘
		for (int i = 0; i < (x * W).rows(); i++)
		{
			result.row(i) = result.row(i) + b;//使每一层与偏置相加
		}
		return result;
	}
	MatrixXd backward(MatrixXd dout) {
		MatrixXd dx = dout * (W.transpose());//求出了正向传播时x的导数
		dW = (x.transpose()) * dout;//求出了正向传播时权重的导数
		db = dout.colwise().sum();//求出了正向传播时偏置的导数
		return dx;
	}
};
class SoftmaxWithLossLayer {//计算误差的层，也是最后一层
public:
	double loss;
	MatrixXd y;//存储了正向传播时的y
	MatrixXd t;//存储了正向传播时的t
	double forward(MatrixXd y1, MatrixXd t1) {
		t = t1;
		y = y1;
		Cross_entropy_error_Layer Cross_entropy_error_Layer;
		loss = Cross_entropy_error_Layer.forward(y, t);//计算误差
		printf("平均误差：%lf\n", loss);
		return loss;
	};
	MatrixXd backward() {
		int batch_size = t.rows();
		return (y - t).array() / batch_size;
	};
};
class Network_2_layer//两层神经网络的架构
{
public:
	map<string, MatrixXd> params;//存储了各层的权重和偏置
	map<string, AffineLayer> Affinelayers;//定义一个存储各Affine层的map
	map<string, ReLULayer> ReLUlayers;//定义一个存储各ReLU层的map
	SoftmaxWithLossLayer LastLayer;//定义了最后一层
	ReLULayer ReLULayer1;//定义了ReLU层
	void init(int input_size, int hidden_size, int output_size, double weight_init_std)//初始化神经网络
	{
		MatrixXd hidden_zeros(1, hidden_size);//生成一个一维矩阵，长度与隐藏层大小相同，将用于存储第一层网络的偏置
		MatrixXd output_zeros(1, output_size);//生成一个一维矩阵，长度与隐藏层大小相同，将用于存储输出层网络的偏置
		params["W1"] = weight_init_std * MatrixXd::Random(input_size, hidden_size);//在{-1,1}内随机生成第一层的权重
		params["b1"] = hidden_zeros.setZero();//初始化第一层网络的偏置，全部赋值为0
		params["W2"] = weight_init_std * MatrixXd::Random(hidden_size, output_size);//在{-1,1}内随机生成第二层的权重
		params["b2"] = output_zeros.setZero();//初始化输出层网络的偏置，全部赋值为0

		AffineLayer AffineLayer1;//定义了Affine层的第一层
		AffineLayer AffineLayer2;//定义了Affine层的第二层
		ReLULayer ReLULayer1;//定义了ReLU层
		Affinelayers["Affine1"] = AffineLayer1;//将Affine层的第一层添加进存储各Affine层的map
		Affinelayers["Affine2"] = AffineLayer2;//将Affine层的第二层添加进存储各Affine层的map
		Affinelayers["Affine1"].init(params["W1"], params["b1"]);//初始化Affine层的第一层
		Affinelayers["Affine2"].init(params["W2"], params["b2"]);//初始化Affine层的第二层
		ReLUlayers["ReLU1"] = ReLULayer1;//初始化ReLU层
	}

	MatrixXd  predict(MatrixXd x) {//神经网络往前传递
		x = Affinelayers["Affine1"].forward(x);
		x = ReLUlayers["ReLU1"].forward(x);
		x = Affinelayers["Affine2"].forward(x);
		return x;
	}
	double  loss(MatrixXd x, MatrixXd t) {//计算损失度和精度
		MatrixXd y = predict(x);
		SoftmaxLayer SoftmaxLayer;
		y = SoftmaxLayer.forward(y);
		printf("精度:%lf\t", accuracy(y, t));
		return LastLayer.forward(y, t);
	}
	double accuracy(MatrixXd y, MatrixXd t) {//计算精度
		double correct = 0;
		for (int i = 0; i < y.rows(); i++)
		{
			int r, c;
			int r2, c2;
			y.row(i).maxCoeff(&r, &c);
			t.row(i).maxCoeff(&r2, &c2);
			if (c == c2)
				correct = correct + 1.0;
		}
		return  (correct / y.rows());
	}
	grads gradient(MatrixXd x, MatrixXd t)//误差反向传播法的核心
	{
		loss(x, t);//使神经网络往前运行
		MatrixXd dout = LastLayer.backward();//获取输出层的导数
		dout = Affinelayers["Affine2"].backward(dout);//获取Affine层的第二层的导数
		dout = ReLUlayers["ReLU1"].backward(dout);//获取ReLU层的导数
		dout = Affinelayers["Affine1"].backward(dout);//获取Affine层的第一层的导数
		//通过以上反向传播，获取到了输入层的导数
		grads grads;
		//声明了一个存储各层偏导数的struct
		grads.W1 = Affinelayers["Affine1"].dW;//获取第一层神经元的权重的偏导数
		grads.b1 = Affinelayers["Affine1"].db;//获取第一层神经元的偏重的偏导数
		grads.W2 = Affinelayers["Affine2"].dW;//获取第二层神经元的权重的偏导数
		grads.b2 = Affinelayers["Affine2"].db;//获取第二层神经元的偏重的偏导数
		return grads;
	}
};
int ReverseInt(int i)//反转整型数字
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void read_Mnist_Label(string filename, vector<double>& labels)//读取Mnist手写集标签
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((double)label);
		}
	}
}
void read_Mnist_Images(string filename, vector<vector<double>>& images)//读取Mnist手写集图片
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < number_of_images; i++)
		{
			vector<double>tp;


			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));

					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
}
void run_deep_learning(MatrixXd images, MatrixXd labels,int input_size, int hidden_size, int output_size, double weight_init_std, int mini_batch_count, double learning_rate,int learning_times) {
	Network_2_layer network;
	network.init(input_size, hidden_size, output_size, weight_init_std);//初始化网络
	ReLULayer ReLULayer;
	srand(time(nullptr));
	for (int i = 0; i < learning_times; i++) {
		//将传入的原始数据切片
		int randoxNumber = 1 + rand() % (images.rows() - mini_batch_count - 1);
		MatrixXd mini_batch_images(mini_batch_count, images.cols());
		MatrixXd mini_batch_labels(mini_batch_count, output_size);
		for (int t = 0; t < mini_batch_count; t++) {
			for (int z = 0; z < images.cols(); z++) {
				mini_batch_images(t, z) = images(randoxNumber + t, z);
			}
		};
		for (int t = 0; t < mini_batch_count; t++) {
			for (int z = 0; z < output_size; z++) {
				mini_batch_labels(t, z) = labels(randoxNumber + t, z);
			}
		};
		printf("第%d次运算:\t", i);
		grads grads = network.gradient(mini_batch_images, mini_batch_labels);//获取每次误差反向传播后的梯度值
		network.params["W1"] -= learning_rate * grads.W1;//更新第一层神经元的权重
		network.params["b1"] -= learning_rate * grads.b1;//更新第一层神经元的偏置
		network.params["W2"] -= learning_rate * grads.W2;//更新第二层神经元的权重
		network.params["b2"] -= learning_rate * grads.b2;//更新第二层神经元的偏置
		network.Affinelayers["Affine1"].init(network.params["W1"], network.params["b1"]);//更新Affine1层的权重和偏置
		network.Affinelayers["Affine2"].init(network.params["W2"], network.params["b2"]);//更新Affine2层的权重和偏置
	}
}
MatrixXd* read_minst_data() {//读取MNIST手写集数据的函数
	vector<double>labels;
	//read_Mnist_Label("/Users/jonyandunh/Documents/GitHub/BP_OF_DL_CPP/t10k-labels.idx1-ubyte", labels);
	read_Mnist_Label("t10k-labels.idx1-ubyte", labels);
	vector<vector<double>>images;
	//read_Mnist_Images("/Users/jonyandunh/Documents/GitHub/BP_OF_DL_CPP/t10k-images.idx3-ubyte", images);
	read_Mnist_Images("t10k-images.idx3-ubyte", images);
	auto m = images.size();      // 训练集矩阵行数
	auto n = images[0].size();   // 训练集矩阵列数
	auto b = labels.size();      // 训练集标签个数
	MatrixXd images2(images.size(), images[0].size());
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{
			images2(i, j) = images[i][j];
		}
	}
	MatrixXd labels2(images.size(), 10);
	labels2.fill(0);
	for (int j = 0; j < images.size(); j++)
	{
		labels2(j, (int)labels[j]) = 1;
	}
	cout << "训练集矩阵行数:" << m << "\n" << endl;
	cout << "训练集矩阵列数:" << n << "\n" << endl;
	cout << "训练集MatrixXd矩阵行数:" << images2.rows() << "\n" << endl;
	cout << "训练集MatrixXd矩阵列数:" << images2.cols() << "\n" << endl;
	cout << "训练集标签个数:" << labels2.cols() << "\n" << endl;
	cout << "训练集标签MatrixXd矩阵个数:" << b << "\n" << endl;
	static  MatrixXd result[] = { images2,labels2 };
	return result;
}
int main()//主函数
{
	MatrixXd* data = read_minst_data();//读取images\labels的矩阵
	MatrixXd images = data[0];//读取images的矩阵
	MatrixXd labels = data[1];//读取labels的矩阵
	double weight_init_std = 0.01;//初始化权重时的高斯分布规模
	double learning_rate = 0.002;//学习率
	int mini_batch_count = 10;//每一批读取数据的数量
	int learning_times = 10000;//学习次数
	int hidden_size = 100;//隐藏层数量
	int input_size = images.cols();
	int output_size = labels.cols();
	run_deep_learning(images, labels, input_size, hidden_size, output_size, weight_init_std, mini_batch_count, learning_rate, learning_times);
	return 0;
}

