//
//  otsu.cpp
//  ImageProcess
//
//  Created by richard on 17/1/25.
//  Copyright © 2017年 richard. All rights reserved.
//

#include "otsu.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define MAX_GRAY_VALUE 255
#define MIN_GRAY_VALUE 0

using namespace cv;
using namespace std;


int otsu(Mat image, Mat & result)
{
/*
    int i,j;
    int tmp;
    double u0,u1,w0,w1,u,uk;
    double cov;
    double maxcov = 0.0;
    int maxthread = 0;
    
    int hst[MAX_GRAY_VALUE] = {0};
    double pro_hst[MAX_GRAY_VALUE] = {0.0};
    
    int height = dst.cols;
    int width = dst.rows;
    
    //统计每个灰度的数量
    for (i =0 ; i<width; i++) {
        for (j = 0; j<height; j++) {
            tmp = dst.at<uchar>(i,j);
            hst[tmp]++;
            
        }
    }
    //计算每个灰度级占图像中的概率
    for (i = MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++)
        pro_hst[i] = (double)hst[i]/(double)(width*height);
    
    u = 0.0;
    //计算平均灰度
    for (i = MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++)
        u += i*pro_hst[i];
    double det = 0.0;
    for (i = MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++)
        det += (i-u)*(i-u)*pro_hst[i];
    //统计前景和背景的平均灰度值，计算类间方差
    for (i = MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++) {
        w0 = 0.0;
        w1 = 0.0;
        u0 = 0.0;
        u1 = 0.0;
        uk = 0.0;
        
        for (j = MIN_GRAY_VALUE; j<i; j++) {
            uk += j*pro_hst[j];
            w0 += pro_hst[j];
        }
        u0 = uk / w0;
        w1 = 1 - w0;
        u1 = (u - uk)/(1 - w0);
        //计算类间方差
        cov = w0 * w1 * (u1 - u0) * (u1 - u0);
        if (cov > maxcov) {
            maxcov = cov;
            maxthread = i;
        }
    }
 */
    int width = image.cols;
    int height = image.rows;
 //   int x = 0, y = 0;
    int pixelCount[256];
    float pixelPro[256];
    int i, j, pixelSum = width * height, threshold = 0;
    
    uchar* data = (uchar*)image.data;
    
    //初始化
    for (i = 0; i < 256; i++)
    {
        pixelCount[i] = 0;
        pixelPro[i] = 0;
    }
    
    //统计灰度级中每个像素在整幅图像中的个数
    for (i = 0; i < height; i++)
    {
        for (j = 0; j<width;j++)
        {
            pixelCount[data[i * image.step+ j]]++;
        }
    }
    
    
    //计算每个像素在整幅图像中的比例
    for (i = 0; i < 256; i++)
    {
        pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
    }
    
    //经典ostu算法,得到前景和背景的分割
    //遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
    float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
    for (i = 0; i < 256; i++)
    {
        w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
        
        for (j = 0; j < 256; j++)
        {
            if (j <= i) //背景部分
            {
                //以i为阈值分类，第一类总的概率
                w0 += pixelPro[j];
                u0tmp += j * pixelPro[j];
            }
            else       //前景部分
            {
                //以i为阈值分类，第二类总的概率
                w1 += pixelPro[j];
                u1tmp += j * pixelPro[j];
            }
        }
        
        u0 = u0tmp / w0;        //第一类的平均灰度  
        u1 = u1tmp / w1;        //第二类的平均灰度  
        u = u0tmp + u1tmp;      //整幅图像的平均灰度  
        //计算类间方差  
        deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
        //找出最大类间方差以及对应的阈值  
        if (deltaTmp > deltaMax)
        {
            deltaMax = deltaTmp;
            threshold = i;
        }
    }
    
    for (i = 0; i<height; i++)
        for (j = 0; j<width; j++)
            if (image.at<uchar>(i,j) > threshold)
                result.at<uchar>(i,j) = MAX_GRAY_VALUE;
            else
                result.at<uchar>(i,j) = MIN_GRAY_VALUE;
    return threshold;
}
