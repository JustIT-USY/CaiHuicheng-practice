#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const char* imageName_L = "1.jpg"; // 用于检测深度的图像
const char* imageName_R = "26.jpg";
const char* imageList_L = "caliberationpics_L.txt"; // 左相机的标定图片名称列表
const char* imageList_R = "caliberationpics_R.txt"; // 右相机的标定图片名称列表
const char* singleCalibrate_result_L = "calibrationresults_L.txt"; // 存放左相机的标定结果
const char* singleCalibrate_result_R = "calibrationresults_R.txt"; // 存放右相机的标定结果
const char* stereoCalibrate_result_L = "stereocalibrateresult_L.txt"; // 存放立体标定结果
const char* stereoCalibrate_result_R = "stereocalibrateresult_R.txt";
const char* stereoRectifyParams = "stereoRectifyParams.txt"; // 存放立体校正参数
vector<vector<Point2f>> corners_seq_L; // 所有角点坐标
vector<vector<Point2f>> corners_seq_R;
vector<vector<Point3f>> objectPoints_L; // 三维坐标
vector<vector<Point3f>> objectPoints_R;
Mat cameraMatrix_L = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 相机的内参数
Mat cameraMatrix_R = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 初始化相机的内参数
Mat distCoeffs_L = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 相机的畸变系数
Mat distCoeffs_R = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 初始化相机的畸变系数
Mat R, T, E, F; // 立体标定参数
Mat R1, R2, P1, P2, Q; // 立体校正参数
Mat mapl1, mapl2, mapr1, mapr2; // 图像重投影映射表
Mat img1_rectified, img2_rectified, disparity, result3DImage; // 校正图像 视差图 深度图
Size patternSize = Size(9, 6); // 行列内角点个数
Size chessboardSize = Size(30, 30); // 棋盘上每个棋盘格的大小30mm
Size imageSize; // 图像尺寸
Rect validRoi[2];

/*
单目标定
参数：
	imageList		存放标定图片名称的txt
	singleCalibrateResult	存放标定结果的txt
	objectPoints	世界坐标系中点的坐标
	corners_seq		存放图像中的角点,用于立体标定
	cameraMatrix	相机的内参数矩阵
	distCoeffs		相机的畸变系数
	imageSize		输入图像的尺寸（像素）
	patternSize		标定板每行的角点个数, 标定板每列的角点个数 (9, 6)	
	chessboardSize	棋盘上每个方格的边长（mm）
注意：亚像素精确化时，允许输入单通道，8位或者浮点型图像。由于输入图像的类型不同，下面用作标定函数参数的内参数矩阵和畸变系数矩阵在初始化时也要数据注意类型。
*/
bool singleCameraCalibrate(const char* imageList, const char* singleCalibrateResult, vector<vector<Point3f>>& objectPoints, 
	vector<vector<Point2f>>& corners_seq, Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize, Size patternSize, Size chessboardSize)
{
	int n_boards = 0;
	ifstream imageStore(imageList); // 打开存放标定图片名称的txt
	ofstream resultStore(singleCalibrateResult); // 保存标定结果的txt
	// 开始提取角点坐标
	vector<Point2f> corners; // 存放一张图片的角点坐标 
	string imageName; // 读取的标定图片的名称
	while (getline(imageStore, imageName)) // 读取txt的每一行（每一行存放了一张标定图片的名称）
	{
		n_boards++;
		Mat imageInput = imread(imageName);
		cvtColor(imageInput, imageInput, CV_RGB2GRAY);
		imageSize.width = imageInput.cols; // 获取图片的宽度
		imageSize.height = imageInput.rows; // 获取图片的高度
		// 查找标定板的角点
		bool found = findChessboardCorners(imageInput, patternSize, corners); // 最后一个参数int flags的缺省值为：CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
		// 亚像素精确化。在findChessboardCorners中自动调用了cornerSubPix，为了更加精细化，我们自己再调用一次。
		if (found) // 当所有的角点都被找到
		{
			TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001); // 终止标准，迭代40次或者达到0.001的像素精度
			cornerSubPix(imageInput, corners, Size(11, 11), Size(-1, -1), criteria);// 由于我们的图像只存较大，将搜索窗口调大一些，（11， 11）为真实窗口的一半，真实大小为（11*2+1， 11*2+1）--（23， 23）
			corners_seq.push_back(corners); // 存入角点序列
			// 绘制角点
			//drawChessboardCorners(imageInput, patternSize, corners, true);
			//imshow("cornersframe", imageInput);
			//waitKey(500); // 暂停0.5s
		}
	}
	//destroyWindow("cornersframe");
	// 进行相机标定
	// 计算角点对应的三维坐标
	int pic, i, j;
	for (pic = 0; pic < n_boards; pic++)
	{
		vector<Point3f> realPointSet;
		for (i = 0; i < patternSize.height; i++)
		{
			for (j = 0; j < patternSize.width; j++)
			{
				Point3f realPoint;
				// 假设标定板位于世界坐标系Z=0的平面
				realPoint.x = j * chessboardSize.width;
				realPoint.y = i * chessboardSize.height;
				realPoint.z = 0;
				realPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(realPointSet);
	}
	// 执行标定程序
	vector<Mat> rvec; // 旋转向量
	vector<Mat> tvec; // 平移向量
	calibrateCamera(objectPoints, corners_seq, imageSize, cameraMatrix, distCoeffs, rvec, tvec, 0);
	// 保存标定结果
	resultStore << "相机内参数矩阵" << endl;
	resultStore << cameraMatrix << endl << endl;
	resultStore << "相机畸变系数" << endl;
	resultStore << distCoeffs << endl << endl;
	// 计算重投影点，与原图角点比较，得到误差
	double errPerImage = 0.; // 每张图像的误差
	double errAverage = 0.; // 所有图像的平均误差
	double totalErr = 0.; // 误差总和
	vector<Point2f> projectImagePoints; // 重投影点
	for (i = 0; i < n_boards; i++)
	{
		vector<Point3f> tempObjectPoints = objectPoints[i]; // 临时三维点
		// 计算重投影点
		projectPoints(tempObjectPoints, rvec[i], tvec[i], cameraMatrix, distCoeffs, projectImagePoints);
		// 计算新的投影点与旧的投影点之间的误差
		vector<Point2f> tempCornersPoints = corners_seq[i];// 临时存放旧投影点
		Mat tempCornersPointsMat = Mat(1, tempCornersPoints.size(), CV_32FC2); // 定义成两个通道的Mat是为了计算误差
		Mat projectImagePointsMat = Mat(1, projectImagePoints.size(), CV_32FC2);
		// 赋值
		for (int j = 0; j < tempCornersPoints.size(); j++)
		{
			projectImagePointsMat.at<Vec2f>(0, j) = Vec2f(projectImagePoints[j].x, projectImagePoints[j].y);
			tempCornersPointsMat.at<Vec2f>(0, j) = Vec2f(tempCornersPoints[j].x, tempCornersPoints[j].y);
		}
		// opencv里的norm函数其实把这里的两个通道分别分开来计算的(X1-X2)^2的值，然后统一求和，最后进行根号
		errPerImage = norm(tempCornersPointsMat, projectImagePointsMat, NORM_L2) / (patternSize.width * patternSize.height);
		totalErr += errPerImage;
		resultStore << "第" << i + 1 << "张图像的平均误差为：" << errPerImage << endl;
	}
	resultStore << "全局平局误差为：" << totalErr / n_boards << endl;
	imageStore.close();
	resultStore.close();
	return true;
}

/*
双目标定:计算两摄像机相对旋转矩阵 R,平移向量 T, 本征矩阵E, 基础矩阵F
参数：
	stereoCalibrateResult	存放立体标定结果的txt
	objectPoints			三维点
	imagePoints				二维图像上的点
	cameraMatrix			相机内参数
	distCoeffs				相机畸变系数
	imageSize				图像尺寸
	R		左右相机相对的旋转矩阵
	T		左右相机相对的平移向量
	E		本征矩阵
	F		基础矩阵
*/
bool stereoCalibrate(const char* stereoCalibrateResult, vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> imagePoints1, vector<vector<Point2f>> imagePoints2,
	Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size& imageSize, Mat& R, Mat& T, Mat& E, Mat& F)
{
	ofstream stereoStore(stereoCalibrateResult);
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6); // 终止条件
	stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1,
		cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, CALIB_FIX_INTRINSIC, criteria); // 注意参数顺序，可以到保存的文件中查看，避免返回时出错
	stereoStore << "左相机内参数：" << endl;
	stereoStore << cameraMatrix1 << endl;
	stereoStore << "右相机内参数：" << endl;
	stereoStore << cameraMatrix2 << endl;
	stereoStore << "左相机畸变系数：" << endl;
	stereoStore << distCoeffs1 << endl;
	stereoStore << "右相机畸变系数：" << endl;
	stereoStore << distCoeffs2 << endl;
	stereoStore << "旋转矩阵：" << endl;
	stereoStore << R << endl;
	stereoStore << "平移向量：" << endl;
	stereoStore << T << endl;
	stereoStore << "本征矩阵：" << endl;
	stereoStore << E << endl;
	stereoStore << "基础矩阵：" << endl;
	stereoStore << F << endl;
	stereoStore.close(); 
	return true;
}

/*
立体校正
参数：
	stereoRectifyParams	存放立体校正结果的txt
	cameraMatrix			相机内参数
	distCoeffs				相机畸变系数
	imageSize				图像尺寸
	R						左右相机相对的旋转矩阵
	T						左右相机相对的平移向量
	R1, R2					行对齐旋转校正
	P1, P2					左右投影矩阵
	Q						重投影矩阵
	map1, map2				重投影映射表
*/
Rect stereoRectification(const char* stereoRectifyParams, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2,
	Size& imageSize, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2)
{
	Rect validRoi[2];
	ofstream stereoStore(stereoRectifyParams);
	stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
		R, T, R1, R2, P1, P2, Q, 0, -1, imageSize, &validRoi[0], &validRoi[1]);
	// 计算左右图像的重投影映射表
	stereoStore << "R1：" << endl;
	stereoStore << R1 << endl;
	stereoStore << "R2：" << endl;
	stereoStore << R2 << endl;
	stereoStore << "P1：" << endl;
	stereoStore << P1 << endl;
	stereoStore << "P2：" << endl;
	stereoStore << P2 << endl;
	stereoStore << "Q：" << endl;
	stereoStore << Q << endl;
	stereoStore.close();
	cout << "R1:" << endl;
	cout << R1 << endl;
	cout << "R2:" << endl;
	cout << R2 << endl;
	cout << "P1:" << endl;
	cout << P1 << endl;
	cout << "P2:" << endl;
	cout << P2 << endl;
	cout << "Q:" << endl;
	cout << Q << endl;
	initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, mapl1, mapl2);
	initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, mapr1, mapr2);
	return validRoi[0], validRoi[1];
}

/*
计算视差图
参数：
	imageName1	左相机拍摄的图像
	imageName2	右相机拍摄的图像
	img1_rectified	重映射后的左侧相机图像
	img2_rectified	重映射后的右侧相机图像
	map	重投影映射表
*/
bool computeDisparityImage(const char* imageName1, const char* imageName2, Mat& img1_rectified,
	Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], Mat& disparity)
{
	// 首先，对左右相机的两张图片进行重构
	Mat img1 = imread(imageName1);
	Mat img2 = imread(imageName2);
	if (img1.empty() | img2.empty())
	{
		cout << "图像为空" << endl;
	}
	Mat gray_img1, gray_img2;
	cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
	cvtColor(img2, gray_img2, COLOR_BGR2GRAY);
	Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC1); // 注意数据类型
	Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));   
	Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));
	gray_img1.copyTo(canLeft);
	gray_img2.copyTo(canRight);
	imwrite("校正前左右相机图像.jpg", canvas);
	remap(gray_img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
	remap(gray_img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);	
	imwrite("左相机校正图像.jpg", img1_rectified);
	imwrite("右相机校正图像.jpg", img2_rectified);
	img1_rectified.copyTo(canLeft);
	img2_rectified.copyTo(canRight);
	rectangle(canLeft, validRoi[0], Scalar(255, 255, 255), 5, 8);
	rectangle(canRight, validRoi[1], Scalar(255, 255, 255), 5, 8);
	for (int j = 0; j <= canvas.rows; j += 16)
		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
	imwrite("校正后左右相机图像.jpg", canvas);
	// 进行立体匹配
	Ptr<StereoBM> bm = StereoBM::create(16, 9); // Ptr<>是一个智能指针
	bm->compute(img1_rectified, img2_rectified, disparity); // 计算视差图
	disparity.convertTo(disparity, CV_32F, 1.0 / 16);
	// 归一化视差映射
	normalize(disparity, disparity, 0, 256, NORM_MINMAX, -1);
	return true;
}

// 鼠标回调函数，点击视差图显示深度
void onMouse(int event, int x, int y, int flags, void *param)
{
	Point point;
	point.x = x;
	point.y = y;
	if(event == EVENT_LBUTTONDOWN)
	{
		cout << result3DImage.at<Vec3f>(point) << endl;
	}
}

int main()
{
	singleCameraCalibrate(imageList_L, singleCalibrate_result_L, objectPoints_L, corners_seq_L, cameraMatrix_L,
		distCoeffs_L, imageSize, patternSize, chessboardSize);
	cout << "已完成左相机的标定!" << endl;
	singleCameraCalibrate(imageList_R, singleCalibrate_result_R, objectPoints_R, corners_seq_R, cameraMatrix_R,
		distCoeffs_R, imageSize, patternSize, chessboardSize);
	cout << "已完成右相机的标定!" << endl;
	stereoCalibrate(stereoCalibrate_result_L, objectPoints_L, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
		cameraMatrix_R, distCoeffs_R, imageSize, R, T, E, F);
	cout << "相机立体标定完成！" << endl;
	//stereoCalibrate(stereoCalibrate_result_R, objectPoints_R, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
	//	cameraMatrix_R, distCoeffs_R, imageSize, R2, T2, E2, F2);
	//cout << "右相机立体标定完成！" << endl;
	validRoi[0], validRoi[1] = stereoRectification(stereoRectifyParams, cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R,
		imageSize, R, T, R1, R2, P1, P2, Q, mapl1, mapl2, mapr1, mapr2);
	cout << "已创建图像重投影映射表！" << endl;
	computeDisparityImage(imageName_L, imageName_R, img1_rectified, img2_rectified, mapl1, mapl2, mapr1, mapr2, validRoi, disparity);
	cout << "视差图建立完成！" << endl;
	// 从三维投影获得深度映射
	reprojectImageTo3D(disparity, result3DImage, Q);
	imshow("视差图", disparity);
	setMouseCallback("视差图", onMouse);
	waitKey(0);
	//destroyAllWindows();
	return 0;
}
