//
//  EdgeDetect.hpp
//  ImageProcess
//
//  Created by richard on 1/13/17.
//  Copyright © 2017 richard. All rights reserved.
//

#ifndef EdgeDetect_hpp
#define EdgeDetect_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>

void delete_jut(cv::Mat& src, cv::Mat& dst, int uthreshold, int vthreshold, int type);

#endif /* EdgeDetect_hpp */
