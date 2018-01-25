//
//  compensate.cpp
//  ImageProcess
//
//  Created by richard on 17/2/13.
//  Copyright © 2017年 richard. All rights reserved.
//

#include "compensate.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;

void unevenLightCompensate(Mat &image, int blockSize)
{
    if (image.channels() == 3) cvtColor(image, image, 7);
    double average = mean(image)[0];
    int rows_new = ceil(double(image.rows) / double(blockSize));
    int cols_new = ceil(double(image.cols) / double(blockSize));
    Mat blockImage;
    blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
    for (int i = 0; i < rows_new; i++)
    {
        for (int j = 0; j < cols_new; j++)
        {
            int rowmin = i*blockSize;
            int rowmax = (i + 1)*blockSize;
            if (rowmax > image.rows) rowmax = image.rows;
            int colmin = j*blockSize;
            int colmax = (j + 1)*blockSize;
            if (colmax > image.cols) colmax = image.cols;
            Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
            double temaver = mean(imageROI)[0];
            blockImage.at<float>(i, j) = temaver;
        }
    }
    blockImage = blockImage - average;
    Mat blockImage2;
    resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
    Mat image2;
    image.convertTo(image2, CV_32FC1);
    Mat dst = image2 - blockImage2;
    dst.convertTo(image, CV_8UC1);
}