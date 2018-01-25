//
//  ProfilePro.cpp
//  ImageProcess
//
//  Created by richard on 1/14/17.
//  Copyright © 2017 richard. All rights reserved.
//

#include "ProfilePro.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
void RLSA_D_Y(Mat dst,int* y_limit,double threshold_k,int mode)
{
    Mat dstImg ;
    dst.copyTo(dstImg);
    threshold(dstImg,dstImg,150,255,CV_THRESH_BINARY);
    //根据每列的白块间距，确定纵向RLSA的阈值，存入y_limit数组中
    int height = dstImg.rows, width = dstImg.cols;
    int *white_col = new int[height];
    int m = 0 ;
    double tmp1 ;
    double tmp2 ;

    for(int col = 0;col < width;col++)
    {
        tmp1 = 0;
        tmp2 = 0;
        m = 0;
        for (int i = 0; i<height; i++) {
            white_col[i] = 0;
        }
        for (int row = 0; row < height; row++)
        {
            if (dstImg.at<uchar>(row, col) == 255 * mode)
                m++;
            else
            {
                white_col[m]++;
                m=0;
            }
        }
        
        for(int i = 0;i<height;i++)
        {
            tmp1 += white_col[i]*i*i;
            tmp2 += white_col[i];
        }

        if(tmp2!=0)
        {
            y_limit[col] = threshold_k*(sqrt(tmp1/tmp2));
        }
        else
            y_limit[col] = 0;
    }
    
}
void RLSA_D_X(Mat dst,int* x_limit,double threshold_k,int mode)
{
    Mat dstImg ;
    dst.copyTo(dstImg);
    threshold(dstImg,dstImg,150,255,CV_THRESH_BINARY);
    //根据每行的白块间距，确定横向RLSA的阈值，存入x_limit数组中
    int height = dstImg.rows, width = dstImg.cols;
    int *white_row = new int[width];
    int m = 0 ;
    double tmp1 ;
    double tmp2 ;

    for(int row = 0;row < height;row++)
    {
        tmp1 = 0;
        tmp2 = 0;
        m = 0;
        for (int i = 0; i<width; i++) {
            white_row[i] = 0;
        }
        for (int col = 0; col < width; col++)
        {
            
            if (dstImg.at<uchar>(row, col) == 255 * mode)
                m++;
            else
            {
                white_row[m]++;
                m=0;
            }
        }
        for(int i = 0;i<width;i++)
        {
            tmp1 += white_row[i]*i*i;
            tmp2 += white_row[i];
        }
        if(tmp2!=0)
        {
            x_limit[row] =threshold_k*(sqrt(tmp1/tmp2));
        }
        else
            x_limit[row] = 0;
    }
    
 }
int project(const Mat srcImage,int& x_limit,int& y_limit)
{
//    Mat srcImage=imread("test.png");

//    cvtColor(srcImage,srcImage,CV_RGB2GRAY);
//    threshold(srcImage,srcImage,127,255,CV_THRESH_BINARY);
//imshow("d",srcImage);
    Mat dstImg ;

   adaptiveThreshold(srcImage, dstImg, 255, 0, 0, 5, 10);
    int maxLine = 0, maxNum = 0,maxCol=0;//重置255最大数目和最大行
    int minLine = 0, minNum = dstImg.cols,minCol=0,minNum2 = dstImg.rows;//重置255最小数目和最小行
    int height = dstImg.rows, width = dstImg.cols;//图像的高和宽
    int tmp = 0;//保存当前行的255数目
    int meanLine =0;
    int *projArray_x = new int[height];//保存每一行255数目的数组
    int *projArray_y = new int[width];//保存每一列255数目的数组
  //  cv::threshold(dstImg, dstImg, threshold, 255, CV_THRESH_BINARY);//对图形进行二值化处理
    
    //循环访问图像数据，查找每一行的黑点的数目
    for (int i = 0; i < height; ++i)
    {
        tmp = 0;
        for (int j = 0; j < width; ++j)
        {
            if (dstImg.at<uchar>(i, j) == 0)
            {
                ++tmp;
            }
        }
        projArray_x[i] = tmp;
        if (tmp > maxNum)
        {
            maxNum = tmp;
            maxLine = i;
        }
        if (tmp < minNum)
        {
            minNum = tmp;
            minLine = i;
        }
    }

    
    //创建并绘制水平投影图像
    cv::Mat projImg_x(height, width, CV_8U, cv::Scalar(0));
    
    for (int i = 0; i < height; ++i)
    {
        meanLine += projArray_x[i];
        cv::line(projImg_x, cv::Point(width - projArray_x[i], i), cv::Point(width - 1, i), cv::Scalar::all(255));
    }
    
    meanLine /= height;
    x_limit = meanLine / 50;
    imwrite("/Users/richard/Desktop/x.jpg",projImg_x);
    
     maxLine = 0, maxNum = 0,maxCol=0;//重置0最大数目和最大行
     minLine = 0, minNum = dstImg.cols,minCol=0,minNum2 = dstImg.rows;//重置0最小数目和最小行

    //循环访问图像数据，查找每一列的黑点的数目
    for (int col = 0; col < width; ++col)
    {
        tmp = 0;
        for (int row = 0; row < height; ++row)
        {
            if (dstImg.at<uchar>(row, col) == 0)
            {
                tmp++;
            }
        }

        projArray_y[col] = tmp;
        if (tmp > maxNum)
        {
            maxNum = tmp;
            maxCol = col;
        }
        if (tmp < minNum2  )
        {
            minNum2 = tmp;
            minCol = col;
        }
    }
    
    meanLine = 0;
    //创建并绘制垂直投影图像
    cv::Mat projImg_y(height, width,  CV_8U, cv::Scalar(0));
    
    for (int col = 0; col < width; ++col)
    {
        meanLine += projArray_y[col];
        cv::line(projImg_y, cv::Point(col, height - projArray_y[col]), cv::Point(col, height - 1), cv::Scalar::all(255));
    }
    meanLine /= width;
    y_limit = meanLine / 50;
 imwrite("/Users/richard/Desktop/y.jpg",projImg_y);
    delete[] projArray_x;//删除new数组
    delete[] projArray_y;//删除new数组
    
    return 2;
/*
    int *colheight =new int[dstImg.rows];
    memset(colheight,0,dstImg.rows*4);  //数组必须赋初值为零，否则出错。无法遍历数组。
//  memset(colheight,0,src->width*4);
// CvScalar value;
    int value;
    for(int i=0;i<dstImg.cols;i++)
        for(int j=0;j<dstImg.rows;j++)
        {
    //value=cvGet2D(src,j,i);
            value=dstImg.at<uchar>(i,j);
            if(value==255)
            {
                colheight[j]++; //统计每列的白色像素点
            }
    
        }
/*
 for(int i=0;i<dstImg.cols;i++)
 {
 CString str;
 str.Format(TEXT("%d"),colheight[i]);
 MessageBox(str);
 }
 *//*
//dstImg.copyTo(histogramImage);
    Mat histogramImage(dstImg.rows,dstImg.cols,CV_8UC1);
    for(int i=0;i<dstImg.cols;i++)
        for(int j=0;j<dstImg.rows;j++)
        {
            value=0;  //设置为黑色。
            histogramImage.at<uchar>(i,j)=value;
        }
//    imshow("d",histogramImage);
    for(int i=0;i<dstImg.rows;i++)
        for(int j=0;j<colheight[i];j++)
        {
            value=255;  //设置为白色
            histogramImage.at<uchar>(j,i)=value;
        }
//imshow("C",dstImg);
   */
    
}