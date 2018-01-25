//
//  ShowResult.hpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#ifndef ShowResult_hpp
#define ShowResult_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>

using namespace cv;

void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);
#endif /* ShowResult_hpp */
