#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include<string>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>

using namespace cv;
using namespace std;
using namespace rs2;

const int width = 1280;
const int height = 720;
const int fps = 30;
//const int fps = 60;



int main()
{
	//Initialization
	//Depth 
	const char* depth_win = "depth_Image";
	namedWindow(depth_win, WINDOW_AUTOSIZE);
	//IR Left & Right
	const char* left_win = "left_Image";
	namedWindow(left_win, WINDOW_AUTOSIZE);
	const char* right_win = "right_Image";
	namedWindow(right_win, WINDOW_AUTOSIZE);
	//Color
	const char* color_win = "color_Image";
	namedWindow(color_win, WINDOW_AUTOSIZE);

	char LName[100];//left 
	char RName[100];//right 
	char DName[100];//depth
	char CName[100];//color
	long long  i = 0;//counter

	//Pipeline
	rs2::pipeline pipe;
	rs2::config pipe_config;
	pipe_config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
	pipe_config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
	pipe_config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
	pipe_config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);

	rs2::pipeline_profile profile = pipe.start(pipe_config);

	//stream
	auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();

	while (cvGetWindowHandle(depth_win) && cvGetWindowHandle(right_win) && cvGetWindowHandle(left_win) && cvGetWindowHandle(color_win)) // Application still alive?
	{
		//堵塞程序直到新的一帧捕获
		rs2::frameset frameset = pipe.wait_for_frames();
		//取深度图和彩色图
		frame depth_frame = frameset.get_depth_frame();
		video_frame ir_frame_left = frameset.get_infrared_frame(1);
		video_frame ir_frame_right = frameset.get_infrared_frame(2);
		frame color_frame = frameset.get_color_frame();

		Mat dMat_left(Size(width, height), CV_8UC1, (void*)ir_frame_left.get_data());
		Mat dMat_right(Size(width, height), CV_8UC1, (void*)ir_frame_right.get_data());
		Mat depth_image(Size(width, height), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
		Mat color_image(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);

		imshow(left_win, dMat_left);
		imshow(right_win, dMat_right);
		imshow(depth_win, depth_image);
		imshow(color_win, color_image);
		/*waitKey(1);*/
		char c = waitKey(1);
		if (c == 'p')
		{
			sprintf_s(LName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\left_eye\\%d.png", i);
			imwrite(LName, dMat_left);
			sprintf_s(RName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\right_eye\\%d.png", i);
			imwrite(RName, dMat_right);
			sprintf_s(DName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\depth\\%d.png", i);
			imwrite(DName, depth_image);
			sprintf_s(CName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\color\\%d.png", i);
			imwrite(CName, color_image);
			i++;
		}
		else if (c == 'q')
			break;
		/*sprintf_s(LName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\left_eye\\%d.png", i);
		imwrite(LName, dMat_left);
		sprintf_s(RName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\right_eye\\%d.png", i);
		imwrite(RName, dMat_right);
		sprintf_s(DName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\depth\\%d.png", i);
		imwrite(DName, depth_image);
		sprintf_s(CName, "F:\\cpppractice\\D435iimshow\\x64\\Debug\\color\\%d.png", i);
		imwrite(CName, color_image);
		i++;*/
	}
	return 0;
}


