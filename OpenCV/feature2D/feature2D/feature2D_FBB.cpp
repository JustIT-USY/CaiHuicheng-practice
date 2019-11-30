/**
* @概述：采用FAST算子检测特征点，采用BRIEF算子对特征点进行特征提取，并使用BruteForce匹配法进行特征点的匹配
* @类和函数：FastFeatureDetector + BriefDescriptorExtractor + BruteForceMatcher
*
*/

#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>	//SurfFeatureDetector实际在该头文件中
#include <opencv2/legacy/legacy.hpp>	//BruteForceMatcher实际在该头文件中
#include <opencv2/features2d/features2d.hpp>	//FlannBasedMatcher实际在该头文件中
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
	Mat src_1 = imread("color_1.jpg");
	Mat src_2 = imread("color_2.jpg");
	if (!src_1.data || !src_2.data)
	{
		cout << " --(!) Error reading images " << endl;
		return -1;
	}

	//-- Step 1: 使用FAST算法检测特征点
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(20);
	vector<KeyPoint> keypoints_1, keypoints_2;
	fast->detect(src_1, keypoints_1);	//FAST(src_1, keypoints_1, 20); 
	fast->detect(src_2, keypoints_2);	//FAST(src_2, keypoints_2, 20); 
	cout << "img1--number of keypoints: " << keypoints_1.size() << endl;
	cout << "img2--number of keypoints: " << keypoints_2.size() << endl;

	//-- Step 2: 使用BRIEF算法提取特征（计算特征向量）
	BriefDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute(src_1, keypoints_1, descriptors_1);
	extractor.compute(src_2, keypoints_2, descriptors_2);

	//-- Step 3: 使用BruteForce算法法进行暴力匹配
	BruteForceMatcher< L2<float> > matcher;	//FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	cout << "number of matches: " << matches.size() << endl;

	//-- 显示匹配结果
	Mat matchImg;
	drawMatches(src_1, keypoints_1, src_2, keypoints_2, matches, matchImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matching result", matchImg);
	imwrite("match_result.png", matchImg);
	waitKey(0);

	return 0;
}

