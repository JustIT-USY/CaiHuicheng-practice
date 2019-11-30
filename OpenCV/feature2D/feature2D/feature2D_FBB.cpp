/**
* @����������FAST���Ӽ�������㣬����BRIEF���Ӷ����������������ȡ����ʹ��BruteForceƥ�䷨�����������ƥ��
* @��ͺ�����FastFeatureDetector + BriefDescriptorExtractor + BruteForceMatcher
*
*/

#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>	//SurfFeatureDetectorʵ���ڸ�ͷ�ļ���
#include <opencv2/legacy/legacy.hpp>	//BruteForceMatcherʵ���ڸ�ͷ�ļ���
#include <opencv2/features2d/features2d.hpp>	//FlannBasedMatcherʵ���ڸ�ͷ�ļ���
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

	//-- Step 1: ʹ��FAST�㷨���������
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(20);
	vector<KeyPoint> keypoints_1, keypoints_2;
	fast->detect(src_1, keypoints_1);	//FAST(src_1, keypoints_1, 20); 
	fast->detect(src_2, keypoints_2);	//FAST(src_2, keypoints_2, 20); 
	cout << "img1--number of keypoints: " << keypoints_1.size() << endl;
	cout << "img2--number of keypoints: " << keypoints_2.size() << endl;

	//-- Step 2: ʹ��BRIEF�㷨��ȡ��������������������
	BriefDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute(src_1, keypoints_1, descriptors_1);
	extractor.compute(src_2, keypoints_2, descriptors_2);

	//-- Step 3: ʹ��BruteForce�㷨�����б���ƥ��
	BruteForceMatcher< L2<float> > matcher;	//FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	cout << "number of matches: " << matches.size() << endl;

	//-- ��ʾƥ����
	Mat matchImg;
	drawMatches(src_1, keypoints_1, src_2, keypoints_2, matches, matchImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matching result", matchImg);
	imwrite("match_result.png", matchImg);
	waitKey(0);

	return 0;
}

