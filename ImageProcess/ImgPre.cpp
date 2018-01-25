//
//  ImgPre.cpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#include "ImgPre.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <otsu.hpp>
#include <compensate.hpp>
#include <Cluster.hpp>
#include "kht.h"

//Hough变换在倾斜角度大时效果好，倾斜角度小时有过度校正的情况

using namespace cv;
using namespace std;

#define GRAY_THRESH 150
#define HOUGH_VOTE 400

void imgROI( Mat &img,Mat &mask,Mat &roi,int threshold_x,int threshold_y)
{//轮廓裁剪
    int left=img.cols-1,right=0,top=0,down=img.rows-1,count=0;
    int flag = 0;
    for(int i=0;i<img.rows;i++)   //找上邊界
    {
        uchar *ptr_mask = mask.ptr<uchar>(i);
        for(int j=0;j<img.cols;j++)
        {
            if (ptr_mask[j]==0)
            {
                if(i>top) top = i;
                if(i<down) down = i;
                if(j<left) left = j;
                if(j>right) right = j;
            }
        }
    }
    int roi_height = 5-down+top;    //ROI圖像的尺寸
    int roi_widgh = 5+right-left;
    //    cout<<img.rows<<","<<img.cols<<","<<roi_height<<","<<roi_widgh<<endl;
    roi = Mat::zeros(roi_height,roi_widgh,CV_8UC1);
   for(int i=0;i<5;i++)
    {
        for(int j=0;j<roi.cols;j++)
        {
            roi.at<uchar>(i,j) = 255;
        }
    }
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<roi.rows;j++)
        {
            roi.at<uchar>(j,i) = 255;
        }
    }
   
    for(int i=5;i<roi.rows;i++)
    {
        for(int j=5;j<roi.cols;j++)
        {
            if ((i+down)<=top  && (j+left)<=right) {

                roi.at<uchar>(i,j) = img.at<uchar>(i+down, j+left);
            }
            else
                roi.at<uchar>(i,j) = 255;
        }
    }
}


int Imgpre(char* filename, Mat srcImg)
{
    //读入目标图并进行灰度转换
   srcImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if(srcImg.empty())
        return -1;
    //使用对比度补偿对目标图处理
    unevenLightCompensate(srcImg, 10);
    //使用otsu法实现图像二值化
    otsu(srcImg, srcImg);
    //设置中心及初始化各值
    Point center(srcImg.cols/2, srcImg.rows/2);
    Mat padded;
    int opWidth = getOptimalDFTSize(srcImg.rows);
    int opHeight = getOptimalDFTSize(srcImg.cols);
    copyMakeBorder(srcImg, padded, 0, opWidth-srcImg.rows,
                   0, opHeight-srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat comImg;
    //合并
    merge(planes,2,comImg);
    //进行dft变换
    dft(comImg, comImg);
    //计算幅度
    split(comImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    //对数变换
    Mat magMat = planes[0];
    magMat += Scalar::all(1);
    log(magMat, magMat);
    
    //与-2求与，使宽高均为偶数
    magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
    //使原点居中，高频部分移到边缘
    int cx = magMat.cols/2;
    int cy = magMat.rows/2;
    Mat q0(magMat, Rect(0, 0, cx, cy));
    Mat q1(magMat, Rect(0, cy, cx, cy));
    Mat q2(magMat, Rect(cx, cy, cx, cy));
    Mat q3(magMat, Rect(cx, 0, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q2.copyTo(q0);
    tmp.copyTo(q2);
    q1.copyTo(tmp);
    q3.copyTo(q1);
    tmp.copyTo(q3);
    
    //Normalize the magnitude to [0,1], then to[0,255]
    normalize(magMat, magMat, 0, 1, CV_MINMAX);
    Mat magImg(magMat.size(), CV_8UC1);
    magMat.convertTo(magImg,CV_8UC1,255,0);
    //二值化
    threshold(magImg,magImg,GRAY_THRESH,255,CV_THRESH_BINARY);
    //用Hough变换查找直线
    vector<Vec2f> lines;
    float pi180 = (float)CV_PI/180;
    Mat linImg(magImg.size(),CV_8UC3);
    HoughLines(magImg,lines,1,pi180,HOUGH_VOTE,0,0);
    //绘出找到的直线图，并保存
    int numLines = lines.size();
    for(int l=0; l<numLines; l++)
    {
        float rho = lines[l][0], theta = lines[l][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 10000*(-b));
        pt1.y = cvRound(y0 + 10000*(a));
        pt2.x = cvRound(x0 - 10000*(-b));
        pt2.y = cvRound(y0 - 10000*(a));
        line(linImg,pt1,pt2,Scalar(255,0,0),3,8,0);
    }
    imwrite("/Users/richard/Desktop/IMGline.jpg",linImg);

    //找出直线的倾斜角
    float angle=0;
    float piThresh = (float)CV_PI/90;
    float pi2 = CV_PI/2;
    int count = 0;
    for(int l=0; l<numLines; l++)
    {
        float theta = lines[l][1];
        if(abs(theta) < piThresh || abs(theta-pi2) < piThresh)
            continue;
        else{
            count++;
            angle = theta;
            break;
        }
    }
    angle = angle<pi2 ? angle : angle-CV_PI;
    if(angle != pi2){
        float angleT = srcImg.rows*tan(angle)/srcImg.cols;
        angle = atan(angleT);
    }
    float angleD = angle*180/(float)CV_PI;
    cout << "需旋转的角度为:" << endl << angleD << "°" << endl;
    //使用高斯平滑去噪
    GaussianBlur(srcImg, srcImg, Size(3,3),1);
    //求原图外接矩形，并将该外接矩形旋转所求的角度
    Mat rot = cv::getRotationMatrix2D(center, angleD, 1);
    Rect bbox = cv::RotatedRect(center, srcImg.size(), angleD).boundingRect();
    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    cv::Mat dst;
    cv::warpAffine(srcImg, dst, rot, bbox.size(),INTER_LINEAR,BORDER_CONSTANT,cvScalarAll(255));
    imwrite("/Users/richard/Desktop/finImg.jpg",dst);
    
     
    return 0;
  }
int Imgpre_Hough(char* filename, Mat srcImg)
{
    //读入目标图并进行灰度转换
    srcImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if(srcImg.empty())
        return -1;
    
    //使用对比度补偿对目标图处理
    unevenLightCompensate(srcImg, 10);
    //使用otsu法实现图像二值化
    otsu(srcImg, srcImg);
 //   threshold(srcImg,srcImg,GRAY_THRESH,255,CV_THRESH_BINARY);
    
    Mat magImg(srcImg.size(), CV_8UC1);
    srcImg.copyTo(magImg);
    Size size = magImg.size();
    
    vector<Vec4i> lines2;
    
    float pi180 = (float)CV_PI/180;

    adaptiveThreshold(magImg, magImg, 255, 0, cv::THRESH_BINARY_INV, 5, 10);

    dilate(magImg, magImg, Mat(),Point(0,0),10);

    IplImage * RotateRow = cvCreateImage(cvSize(magImg.cols, magImg.rows), 8, 1); //Mat转IplImage
    IplImage ipltemp = magImg;
    cvCopy(&ipltemp, RotateRow);

    Mat drawing = cv::cvarrToMat(RotateRow) ;
    erode(drawing, drawing, Mat(),Point(0,0),15);

    HoughLinesP(drawing, lines2, 1, CV_PI/180, 100, 0, 0);
    imwrite("/Users/richard/Desktop/ImgHough.jpg",drawing);
    
    
    cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
    double angle = 0.;

    int numLines = lines2.size();
    
    float piThresh = (float)CV_PI/90;
    float pi2 = CV_PI/2;

    for(int l=0; l<numLines; l++)
    {
        float theta = lines2[l][1];
        if(abs(theta) < piThresh || abs(theta-pi2) < piThresh)
            continue;
        else{
            angle = theta;
            break;
        }
    }

    angle = angle<pi2 ? angle : angle-CV_PI;
    if(angle != pi2){
        float angelT = srcImg.rows*tan(angle)/srcImg.cols;
        angle = atan(angelT);
    }
    float angelD = angle*180/(float)CV_PI;
    cout << "the rotation angel to be applied:" << endl << angelD << endl << endl;

    //Rotate the image to recover
    double a = sin(angelD), b = cos(angelD);
    int width = srcImg.cols;
    int height = srcImg.rows;
    int width_rotate= int(height * fabs(a) + width * fabs(b));
    int height_rotate=int(width * fabs(a) + height * fabs(b));

    float map[6];
    CvMat map_matrix = cvMat(2, 3, CV_32F, &map);
    // 旋转中心
    CvPoint2D32f center2 = cvPoint2D32f(width / 2, height / 2);
    cv2DRotationMatrix(center2, angelD, 1.0, &map_matrix);
    map[2] += (width_rotate+1 - width) / 2;
    map[5] += (height_rotate+1 - height) / 2;
    IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 1);
   
    IplImage* img = cvCreateImage(cvSize(srcImg.cols, srcImg.rows), 8, 1);
    ipltemp = srcImg;
    cvCopy(&ipltemp, img);
    
    cvWarpAffine( img,img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(255));
    
    cvSaveImage("/Users/richard/Desktop/finImg.jpg", img_rotate);

    return 0;
}
int Imgpre_KHT(char* filename, Mat srcImg)
{
    //读入目标图并进行灰度转换
    srcImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if(srcImg.empty())
        return -1;
    //使用对比度补偿对目标图处理
    unevenLightCompensate(srcImg, 10);
    //使用otsu法实现图像二值化
    otsu(srcImg, srcImg);
    Mat magImg(srcImg.size(), CV_8UC1);
    srcImg.copyTo(magImg);
    Size size = magImg.size();
    vector<Vec4i> lines2;
    
    float pi180 = (float)CV_PI/180;
    
    vector<vector<Point> > contours;//存储轮廓的点集合
    vector<Vec4i> hierarchy;//构建轮廓的层次结构

    
    adaptiveThreshold(magImg, magImg, 255, 0, cv::THRESH_BINARY_INV, 5, 10);
    
    findContours( magImg, contours, hierarchy,
                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );   /// 对每个轮廓计算其凸包
    vector<Rect> boundRect( contours.size() );

    Scalar color = Scalar( 255,255,0 );
     vector<vector<Point> >hull( contours.size() );
    
    Mat drawing = Mat::zeros( magImg.size(), CV_8UC1 );
    
    for( int i = 0; i < contours.size(); i++ )
    {
        convexHull( Mat(contours[i]), hull[i], false );//凸包计算
    }
    
    for( int i = 0; i< contours.size(); i++ )
    {
        boundRect[i] = boundingRect( Mat(contours[i]) );
        double area = contourArea(contours[i]);
        double rate = area/boundRect[i].width*boundRect[i].height;
        if (rate>0.06 && area<150 )
        {
             drawContours( drawing, hull, i,
                          color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );//画凸包
        }
    }
    imwrite("/Users/richard/Desktop/2.jpg", drawing );
    
    dilate(drawing, drawing, Mat(),Point(0,0),15);
    erode(drawing, drawing, Mat(),Point(0,0),14);
    
    imwrite("/Users/richard/Desktop/1.jpg", drawing );
    
    boundRect.clear();
    vector <Rect>().swap(boundRect);
    hull.clear();
    contours.clear();
    hierarchy.clear();

    Mat drawing2 = drawing.clone();
    static lines_list_t lines3;
    
    kht(lines3, drawing2.data, drawing2.cols,
        drawing2.rows,20,1.0,1.0,1.0,1.0);

    
    cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
    double angle = 0.0;
    
    int numLines = lines3.size();
    
    float piThresh = (float)CV_PI/90;
    float pi2 = CV_PI/2;
    
    for(int l=0; l<numLines; l++)
    {
        float theta = lines3[l].theta;
        if(abs(theta) < piThresh || abs(theta-pi2) < piThresh)
            continue;
        else{
            angle = theta;
            break;
        }
    }
    
    angle = angle<pi2 ? angle : angle-CV_PI;
    if(angle != pi2){
        float angelT = srcImg.rows*tan(angle)/srcImg.cols;
        angle = atan(angelT);
    }
    float angelD = angle*180/(float)CV_PI;
    cout << "the rotation angel to be applied:" << endl << angelD << endl << endl;
    
    //Rotate the image to recover
    double a = sin(angelD), b = cos(angelD);
    int width = srcImg.cols;
    int height = srcImg.rows;
    int width_rotate= int(height * fabs(a) + width * fabs(b));
    int height_rotate=int(width * fabs(a) + height * fabs(b));
    
    float map[6];
    CvMat map_matrix = cvMat(2, 3, CV_32F, &map);
    // 旋转中心
    CvPoint2D32f center2 = cvPoint2D32f(width / 2, height / 2);
    cv2DRotationMatrix(center2, angelD, 1.0, &map_matrix);
    map[2] += (width_rotate+1 - width) / 2;
    map[5] += (height_rotate+1 - height) / 2;
    IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 1);
    
    IplImage* img = cvCreateImage(cvSize(srcImg.cols, srcImg.rows), 8, 1);
    IplImage ipltemp = srcImg;
    cvCopy(&ipltemp, img);
    
    cvWarpAffine( img,img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(255));
    
    cvSaveImage("/Users/richard/Desktop/finImg.jpg", img_rotate);
    
    return 0;
  }

void horizontalRLSA(Mat &input, Mat &output, int* thresh,int mode )
{
    threshold(input,input,150,255,CV_THRESH_BINARY);
    threshold(output,output,150,255,CV_THRESH_BINARY);
    int count = 0;
    int flag = 0;
    int th = 255*(1-mode);
    for (int j = 0; j < input.rows; j++)
    {
        flag = 0;
        count = 0;
        for (int i = 0; i < input.cols; i++)
        {
            if (input.at<uchar>(j, i) == 255 * mode)
            {
                flag = 255;
                count++;
            }
            else
            {
                if (flag == 255 && count <= thresh[j])
                {
                    output(Rect(i - count, j, count, 1)).setTo(Scalar::all(th));
                }
                flag = 0;
                count = 0;
            }
        }
    }
}
void verticalRLSA(Mat& input, Mat &output, int* thresh,int mode )
{

    threshold(input,input,150,255,CV_THRESH_BINARY);
    threshold(output,output,150,255,CV_THRESH_BINARY);
 //   for (int i = 0; i < input.cols; i++)
 //   {
    int count=0 ;
    int flag=0 ;
    int th = 255*(1-mode);
    for(int i = 0; i<input.cols;i++)
    {
        flag = 0;
        count = 0;
        for (int j = 0; j < input.rows; j++)
        {
            if (input.at<uchar>(j, i) == 255 * mode)
            {
                flag = 255;
                count++;
            }
            else
            {
                if (flag == 255 && count <= thresh[i])
                {
                    output(Rect(i, j - count, 1, count)).setTo(Scalar::all(th));
                }
                flag = 0;
                count = 0;
            }
       }
    }
   // }
}