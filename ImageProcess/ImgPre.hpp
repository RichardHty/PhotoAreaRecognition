//
//  ImgPre.hpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#ifndef ImgPre_hpp
#define ImgPre_hpp

#include <opencv2/core/core.hpp>
int Imgpre (char* filename,  cv::Mat srcImg);
void imgROI( cv::Mat &img,cv::Mat &mask,cv::Mat &roi,int threshold_x,int threshold_y);
void horizontalRLSA(cv::Mat &input, cv::Mat &output, int* thresh,int mode =0);
void verticalRLSA(cv::Mat &input, cv::Mat &output, int* thresh,int mode = 0);
int Imgpre_Hough(char* filename, cv::Mat srcImg);
int Imgpre_KHT(char* filename, cv::Mat srcImg);

#endif /* ImgPre_hpp */
