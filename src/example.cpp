#include <iostream>
#include <vector>
#include <string>

#include "fhog.h"
#include "fhog1.hpp"

#include <opencv2\opencv.hpp>

using namespace std;

int main(int argc, char** argv)
{
	string s = "F:\\Pictures\\2_small.mp4_0001.jpg";

    cv::Mat img = cv::imread(s);

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

	auto st_new_fhog = cv::getTickCount();

	FHOG fhogDescripter;

	for (int i = 0; i < 200; ++i)
	{
		fhogDescripter.static_Init(img.size(), 4);

		cv::Mat feat;

		fhogDescripter.compute(img, feat, 4, 0.2);
	}
    
	auto ed_new_fhog = cv::getTickCount();

	double time_new_fhog = (ed_new_fhog - st_new_fhog) / cv::getTickFrequency();


	IplImage img_0 = img;

	auto st_old_fhog = cv::getTickCount();
	CvLSVMFeatureMapCaskade** map = new CvLSVMFeatureMapCaskade*;
	for (int i = 0; i < 200; ++i)
	{
		getFeatureMaps(&img_0, 4, map);
		normalizeAndTruncate(*map, 0.2);
		PCAFeatureMaps(*map);

		freeFeatureMapObject(map);
	}

	auto ed_old_fhog = cv::getTickCount();

	double time_old_fhog = (ed_old_fhog - st_old_fhog) / cv::getTickFrequency();

	cout << "Original FHOG:" << time_old_fhog << endl;
	cout << "New FHOG:" << time_new_fhog << endl;

	cv::imshow("image", img);
	cv::waitKey(0);

	delete map;

    return 0;
}