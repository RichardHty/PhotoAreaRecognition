//
//  compensate.hpp
//  ImageProcess
//
//  Created by richard on 17/2/13.
//  Copyright © 2017年 richard. All rights reserved.
//

#ifndef compensate_hpp
#define compensate_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
void unevenLightCompensate(cv::Mat &image, int blockSize);
#endif /* compensate_hpp */
