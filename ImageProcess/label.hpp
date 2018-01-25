//
//  label.hpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#ifndef label_hpp
#define label_hpp
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
//void  yuBwLabel( const cv::Mat &bwImg, cv::Mat &labImg );
void  yuBwLabel( const cv::Mat &_binImg, cv::Mat & _lableImg );
int label (int *mask, IplImage *img1);
void getConnectedDomain(cv::Mat& src, std::vector<cv::Rect>& boundingbox);

#endif /* label_hpp */
