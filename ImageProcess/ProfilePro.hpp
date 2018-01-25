//
//  ProfilePro.hpp
//  ImageProcess
//
//  Created by richard on 1/14/17.
//  Copyright © 2017 richard. All rights reserved.
//

#ifndef ProfilePro_hpp
#define ProfilePro_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
int project(const cv::Mat srcImage,int& x_limit,int& y_limit);
void RLSA_D_Y(cv::Mat dst,int* y_limit,double threshold_k,int mode = 0);
void RLSA_D_X(cv::Mat dst,int* x_limit,double threshold_k,int mode = 0);
#endif /* ProfilePro_hpp */
