#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const char* imageName_L = "1.jpg"; // ���ڼ����ȵ�ͼ��
const char* imageName_R = "26.jpg";
const char* imageList_L = "caliberationpics_L.txt"; // ������ı궨ͼƬ�����б�
const char* imageList_R = "caliberationpics_R.txt"; // ������ı궨ͼƬ�����б�
const char* singleCalibrate_result_L = "calibrationresults_L.txt"; // ���������ı궨���
const char* singleCalibrate_result_R = "calibrationresults_R.txt"; // ���������ı궨���
const char* stereoCalibrate_result_L = "stereocalibrateresult_L.txt"; // �������궨���
const char* stereoCalibrate_result_R = "stereocalibrateresult_R.txt";
const char* stereoRectifyParams = "stereoRectifyParams.txt"; // �������У������
vector<vector<Point2f>> corners_seq_L; // ���нǵ�����
vector<vector<Point2f>> corners_seq_R;
vector<vector<Point3f>> objectPoints_L; // ��ά����
vector<vector<Point3f>> objectPoints_R;
Mat cameraMatrix_L = Mat(3, 3, CV_32FC1, Scalar::all(0)); // ������ڲ���
Mat cameraMatrix_R = Mat(3, 3, CV_32FC1, Scalar::all(0)); // ��ʼ��������ڲ���
Mat distCoeffs_L = Mat(1, 5, CV_32FC1, Scalar::all(0)); // ����Ļ���ϵ��
Mat distCoeffs_R = Mat(1, 5, CV_32FC1, Scalar::all(0)); // ��ʼ������Ļ���ϵ��
Mat R, T, E, F; // ����궨����
Mat R1, R2, P1, P2, Q; // ����У������
Mat mapl1, mapl2, mapr1, mapr2; // ͼ����ͶӰӳ���
Mat img1_rectified, img2_rectified, disparity, result3DImage; // У��ͼ�� �Ӳ�ͼ ���ͼ
Size patternSize = Size(9, 6); // �����ڽǵ����
Size chessboardSize = Size(30, 30); // ������ÿ�����̸�Ĵ�С30mm
Size imageSize; // ͼ��ߴ�
Rect validRoi[2];

/*
��Ŀ�궨
������
	imageList		��ű궨ͼƬ���Ƶ�txt
	singleCalibrateResult	��ű궨�����txt
	objectPoints	��������ϵ�е������
	corners_seq		���ͼ���еĽǵ�,��������궨
	cameraMatrix	������ڲ�������
	distCoeffs		����Ļ���ϵ��
	imageSize		����ͼ��ĳߴ磨���أ�
	patternSize		�궨��ÿ�еĽǵ����, �궨��ÿ�еĽǵ���� (9, 6)	
	chessboardSize	������ÿ������ı߳���mm��
ע�⣺�����ؾ�ȷ��ʱ���������뵥ͨ����8λ���߸�����ͼ����������ͼ������Ͳ�ͬ�����������궨�����������ڲ�������ͻ���ϵ�������ڳ�ʼ��ʱҲҪ����ע�����͡�
*/
bool singleCameraCalibrate(const char* imageList, const char* singleCalibrateResult, vector<vector<Point3f>>& objectPoints, 
	vector<vector<Point2f>>& corners_seq, Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize, Size patternSize, Size chessboardSize)
{
	int n_boards = 0;
	ifstream imageStore(imageList); // �򿪴�ű궨ͼƬ���Ƶ�txt
	ofstream resultStore(singleCalibrateResult); // ����궨�����txt
	// ��ʼ��ȡ�ǵ�����
	vector<Point2f> corners; // ���һ��ͼƬ�Ľǵ����� 
	string imageName; // ��ȡ�ı궨ͼƬ������
	while (getline(imageStore, imageName)) // ��ȡtxt��ÿһ�У�ÿһ�д����һ�ű궨ͼƬ�����ƣ�
	{
		n_boards++;
		Mat imageInput = imread(imageName);
		cvtColor(imageInput, imageInput, CV_RGB2GRAY);
		imageSize.width = imageInput.cols; // ��ȡͼƬ�Ŀ��
		imageSize.height = imageInput.rows; // ��ȡͼƬ�ĸ߶�
		// ���ұ궨��Ľǵ�
		bool found = findChessboardCorners(imageInput, patternSize, corners); // ���һ������int flags��ȱʡֵΪ��CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
		// �����ؾ�ȷ������findChessboardCorners���Զ�������cornerSubPix��Ϊ�˸��Ӿ�ϸ���������Լ��ٵ���һ�Ρ�
		if (found) // �����еĽǵ㶼���ҵ�
		{
			TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001); // ��ֹ��׼������40�λ��ߴﵽ0.001�����ؾ���
			cornerSubPix(imageInput, corners, Size(11, 11), Size(-1, -1), criteria);// �������ǵ�ͼ��ֻ��ϴ󣬽��������ڵ���һЩ����11�� 11��Ϊ��ʵ���ڵ�һ�룬��ʵ��СΪ��11*2+1�� 11*2+1��--��23�� 23��
			corners_seq.push_back(corners); // ����ǵ�����
			// ���ƽǵ�
			//drawChessboardCorners(imageInput, patternSize, corners, true);
			//imshow("cornersframe", imageInput);
			//waitKey(500); // ��ͣ0.5s
		}
	}
	//destroyWindow("cornersframe");
	// ��������궨
	// ����ǵ��Ӧ����ά����
	int pic, i, j;
	for (pic = 0; pic < n_boards; pic++)
	{
		vector<Point3f> realPointSet;
		for (i = 0; i < patternSize.height; i++)
		{
			for (j = 0; j < patternSize.width; j++)
			{
				Point3f realPoint;
				// ����궨��λ����������ϵZ=0��ƽ��
				realPoint.x = j * chessboardSize.width;
				realPoint.y = i * chessboardSize.height;
				realPoint.z = 0;
				realPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(realPointSet);
	}
	// ִ�б궨����
	vector<Mat> rvec; // ��ת����
	vector<Mat> tvec; // ƽ������
	calibrateCamera(objectPoints, corners_seq, imageSize, cameraMatrix, distCoeffs, rvec, tvec, 0);
	// ����궨���
	resultStore << "����ڲ�������" << endl;
	resultStore << cameraMatrix << endl << endl;
	resultStore << "�������ϵ��" << endl;
	resultStore << distCoeffs << endl << endl;
	// ������ͶӰ�㣬��ԭͼ�ǵ�Ƚϣ��õ����
	double errPerImage = 0.; // ÿ��ͼ������
	double errAverage = 0.; // ����ͼ���ƽ�����
	double totalErr = 0.; // ����ܺ�
	vector<Point2f> projectImagePoints; // ��ͶӰ��
	for (i = 0; i < n_boards; i++)
	{
		vector<Point3f> tempObjectPoints = objectPoints[i]; // ��ʱ��ά��
		// ������ͶӰ��
		projectPoints(tempObjectPoints, rvec[i], tvec[i], cameraMatrix, distCoeffs, projectImagePoints);
		// �����µ�ͶӰ����ɵ�ͶӰ��֮������
		vector<Point2f> tempCornersPoints = corners_seq[i];// ��ʱ��ž�ͶӰ��
		Mat tempCornersPointsMat = Mat(1, tempCornersPoints.size(), CV_32FC2); // ���������ͨ����Mat��Ϊ�˼������
		Mat projectImagePointsMat = Mat(1, projectImagePoints.size(), CV_32FC2);
		// ��ֵ
		for (int j = 0; j < tempCornersPoints.size(); j++)
		{
			projectImagePointsMat.at<Vec2f>(0, j) = Vec2f(projectImagePoints[j].x, projectImagePoints[j].y);
			tempCornersPointsMat.at<Vec2f>(0, j) = Vec2f(tempCornersPoints[j].x, tempCornersPoints[j].y);
		}
		// opencv���norm������ʵ�����������ͨ���ֱ�ֿ��������(X1-X2)^2��ֵ��Ȼ��ͳһ��ͣ������и���
		errPerImage = norm(tempCornersPointsMat, projectImagePointsMat, NORM_L2) / (patternSize.width * patternSize.height);
		totalErr += errPerImage;
		resultStore << "��" << i + 1 << "��ͼ���ƽ�����Ϊ��" << errPerImage << endl;
	}
	resultStore << "ȫ��ƽ�����Ϊ��" << totalErr / n_boards << endl;
	imageStore.close();
	resultStore.close();
	return true;
}

/*
˫Ŀ�궨:����������������ת���� R,ƽ������ T, ��������E, ��������F
������
	stereoCalibrateResult	�������궨�����txt
	objectPoints			��ά��
	imagePoints				��άͼ���ϵĵ�
	cameraMatrix			����ڲ���
	distCoeffs				�������ϵ��
	imageSize				ͼ��ߴ�
	R		���������Ե���ת����
	T		���������Ե�ƽ������
	E		��������
	F		��������
*/
bool stereoCalibrate(const char* stereoCalibrateResult, vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> imagePoints1, vector<vector<Point2f>> imagePoints2,
	Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size& imageSize, Mat& R, Mat& T, Mat& E, Mat& F)
{
	ofstream stereoStore(stereoCalibrateResult);
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6); // ��ֹ����
	stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1,
		cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, CALIB_FIX_INTRINSIC, criteria); // ע�����˳�򣬿��Ե�������ļ��в鿴�����ⷵ��ʱ����
	stereoStore << "������ڲ�����" << endl;
	stereoStore << cameraMatrix1 << endl;
	stereoStore << "������ڲ�����" << endl;
	stereoStore << cameraMatrix2 << endl;
	stereoStore << "���������ϵ����" << endl;
	stereoStore << distCoeffs1 << endl;
	stereoStore << "���������ϵ����" << endl;
	stereoStore << distCoeffs2 << endl;
	stereoStore << "��ת����" << endl;
	stereoStore << R << endl;
	stereoStore << "ƽ��������" << endl;
	stereoStore << T << endl;
	stereoStore << "��������" << endl;
	stereoStore << E << endl;
	stereoStore << "��������" << endl;
	stereoStore << F << endl;
	stereoStore.close(); 
	return true;
}

/*
����У��
������
	stereoRectifyParams	�������У�������txt
	cameraMatrix			����ڲ���
	distCoeffs				�������ϵ��
	imageSize				ͼ��ߴ�
	R						���������Ե���ת����
	T						���������Ե�ƽ������
	R1, R2					�ж�����תУ��
	P1, P2					����ͶӰ����
	Q						��ͶӰ����
	map1, map2				��ͶӰӳ���
*/
Rect stereoRectification(const char* stereoRectifyParams, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2,
	Size& imageSize, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2)
{
	Rect validRoi[2];
	ofstream stereoStore(stereoRectifyParams);
	stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
		R, T, R1, R2, P1, P2, Q, 0, -1, imageSize, &validRoi[0], &validRoi[1]);
	// ��������ͼ�����ͶӰӳ���
	stereoStore << "R1��" << endl;
	stereoStore << R1 << endl;
	stereoStore << "R2��" << endl;
	stereoStore << R2 << endl;
	stereoStore << "P1��" << endl;
	stereoStore << P1 << endl;
	stereoStore << "P2��" << endl;
	stereoStore << P2 << endl;
	stereoStore << "Q��" << endl;
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
�����Ӳ�ͼ
������
	imageName1	����������ͼ��
	imageName2	����������ͼ��
	img1_rectified	��ӳ����������ͼ��
	img2_rectified	��ӳ�����Ҳ����ͼ��
	map	��ͶӰӳ���
*/
bool computeDisparityImage(const char* imageName1, const char* imageName2, Mat& img1_rectified,
	Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], Mat& disparity)
{
	// ���ȣ����������������ͼƬ�����ع�
	Mat img1 = imread(imageName1);
	Mat img2 = imread(imageName2);
	if (img1.empty() | img2.empty())
	{
		cout << "ͼ��Ϊ��" << endl;
	}
	Mat gray_img1, gray_img2;
	cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
	cvtColor(img2, gray_img2, COLOR_BGR2GRAY);
	Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC1); // ע����������
	Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));   
	Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));
	gray_img1.copyTo(canLeft);
	gray_img2.copyTo(canRight);
	imwrite("У��ǰ�������ͼ��.jpg", canvas);
	remap(gray_img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
	remap(gray_img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);	
	imwrite("�����У��ͼ��.jpg", img1_rectified);
	imwrite("�����У��ͼ��.jpg", img2_rectified);
	img1_rectified.copyTo(canLeft);
	img2_rectified.copyTo(canRight);
	rectangle(canLeft, validRoi[0], Scalar(255, 255, 255), 5, 8);
	rectangle(canRight, validRoi[1], Scalar(255, 255, 255), 5, 8);
	for (int j = 0; j <= canvas.rows; j += 16)
		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
	imwrite("У�����������ͼ��.jpg", canvas);
	// ��������ƥ��
	Ptr<StereoBM> bm = StereoBM::create(16, 9); // Ptr<>��һ������ָ��
	bm->compute(img1_rectified, img2_rectified, disparity); // �����Ӳ�ͼ
	disparity.convertTo(disparity, CV_32F, 1.0 / 16);
	// ��һ���Ӳ�ӳ��
	normalize(disparity, disparity, 0, 256, NORM_MINMAX, -1);
	return true;
}

// ���ص�����������Ӳ�ͼ��ʾ���
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
	cout << "�����������ı궨!" << endl;
	singleCameraCalibrate(imageList_R, singleCalibrate_result_R, objectPoints_R, corners_seq_R, cameraMatrix_R,
		distCoeffs_R, imageSize, patternSize, chessboardSize);
	cout << "�����������ı궨!" << endl;
	stereoCalibrate(stereoCalibrate_result_L, objectPoints_L, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
		cameraMatrix_R, distCoeffs_R, imageSize, R, T, E, F);
	cout << "�������궨��ɣ�" << endl;
	//stereoCalibrate(stereoCalibrate_result_R, objectPoints_R, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
	//	cameraMatrix_R, distCoeffs_R, imageSize, R2, T2, E2, F2);
	//cout << "���������궨��ɣ�" << endl;
	validRoi[0], validRoi[1] = stereoRectification(stereoRectifyParams, cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R,
		imageSize, R, T, R1, R2, P1, P2, Q, mapl1, mapl2, mapr1, mapr2);
	cout << "�Ѵ���ͼ����ͶӰӳ���" << endl;
	computeDisparityImage(imageName_L, imageName_R, img1_rectified, img2_rectified, mapl1, mapl2, mapr1, mapr2, validRoi, disparity);
	cout << "�Ӳ�ͼ������ɣ�" << endl;
	// ����άͶӰ������ӳ��
	reprojectImageTo3D(disparity, result3DImage, Q);
	imshow("�Ӳ�ͼ", disparity);
	setMouseCallback("�Ӳ�ͼ", onMouse);
	waitKey(0);
	//destroyAllWindows();
	return 0;
}
