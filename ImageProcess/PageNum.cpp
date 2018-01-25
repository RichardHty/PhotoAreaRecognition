//
//  PageNum.cpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#include "PageNum.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <EdgeDetect.hpp>
#include <ImgPre.hpp>
#include <ProfilePro.hpp>

using namespace std;
using namespace cv;
double minarea=20.0;
double tmparea = 0.0;

bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
{
    if ( rectA.x > rectB.x + rectB.width ) { return false; }
    if ( rectA.y > rectB.y + rectB.height ) { return false; }
    if ( (rectA.x + rectA.width) < rectB.x ) { return false; }
    if ( (rectA.y + rectA.height) < rectB.y ) { return false; }
    
    float colInt =  fmin(rectA.x+rectA.width,rectB.x+rectB.width) - fmax(rectA.x, rectB.x);
    float rowInt =  fmin(rectA.y+rectA.height,rectB.y+rectB.height) - fmax(rectA.y,rectB.y);
    float intersection = colInt * rowInt;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float intersectionPercent =  intersection / (areaA + areaB - intersection);
    
    if (( (0 < intersectionPercent)&&(intersectionPercent < 1)&&(intersection != areaA) && (intersection != areaB))||intersectionPercent==1)
    {
        return true;
    }
    
 /*原
    if ( (0 < intersectionPercent)&&(intersectionPercent < 1)&&(intersection != areaA)&&(intersection != areaB) )
    {
        return true;
    }
*/
    return false;
}
int pix_num(Mat dst,Rect a )
{

    
    int x1 = a.tl().x;
    int y1 = a.tl().y;
    
    int x4 = a.br().x;
    int y4 = a.br().y;
    
    int count1 = 0;
    int count2 = 0;
    double sum = 0;
    
    int i;
    int j;
    for ( i = y1; i<y4+1; i++) {
        for ( j = x1; j<x4+1; j++) {

            sum += dst.at<uchar>(i, j);


        }
    }

    sum /= 255;

    return sum;
}
double density(int width, int height,double area)
{
    double dens = area / (width*height);
    return dens;
}
double HWRate(int width, int height)
{
    //double hw = min(width,height)/max(width,height);
    double hw = 1.0 * min(width,height)/max(width,height);
    return hw;
}
int dist_center(Rect a,Rect b)
{//矩形中心距
    int distant_x = a.x+a.width/2 - b.x-b.width/2;
    int distant_y = a.y+a.height/2 - b.y-b.height/2;
    int dista = sqrt(distant_x*distant_x + distant_y*distant_y);
    
    return dista;
}
int dist_x(Rect a,Rect b)
{//矩形字间距
    int distant_x = a.x - b.x;
    int dista ;
    if (distant_x<0) {
        if (a.x+a.width>b.x)
        {
                dista = abs(distant_x);
        }
        else
            dista = abs(distant_x) - a.width;
    }
    else
    {
        if (b.x+b.width>a.x)
        {
            dista = distant_x;
        }
        else
            dista = distant_x - b.width;
    }
    return dista;
}
int dist_y(Rect a,Rect b)
{//矩形行间距
    int distant_y = a.y - b.y;
    int dista ;
    if (distant_y<0) {
        if (a.y+a.height>b.y)
        {
            dista = abs(distant_y);
        }
        else
            dista = abs(distant_y) - a.height;
    }
    else
    {
        if (b.y+b.height>a.y)
        {
            dista = distant_y;
        }
        else
            dista = distant_y - b.height;
    }
    return dista;
}
int cmp(const void* p1, const void* p2)
{
    return *((int*)p2)>*((int*)p1)?-1:1;
}
Point min_tl(Rect a,Rect b)
{
    Point fin;
    fin.x = max(a.tl().x,b.tl().x);
    fin.y = min(a.br().y,b.br().y);
    return fin;
}
Point max_br(Rect a,Rect b)
{
    Point fin;
    fin.x = min(a.br().x,b.br().x);
    fin.y = max(a.tl().y,b.tl().y);
    return fin;
}
void connect_words(Mat &drawing,vector<vector<Point> > contours,vector<Vec4i> hierarchy,int times = 1,int dis_x = 10 ,int dis_x_y = 3 ,int dis_y = 2 ,int dis_y_x = 3)
{
    Scalar color = Scalar( 255,255,255 );
    for(int i= 0;i<times;i++)
    {
        findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<Rect> boundRect_second( contours.size() );
    
        for(int i = 0; i< contours.size(); i++)
        {
        //   drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );//画轮廓
        
            boundRect_second[i] = boundingRect( Mat(contours[i]) );
        }
    
        for(int i = 0;i< contours.size();i++)
        {
            for(int j = 0;j<contours.size();j++)
            {
            
                if ((dist_x(boundRect_second[i], boundRect_second[j])<dis_x && dist_y(boundRect_second[i], boundRect_second[j])<dis_x_y) || ( dist_y(boundRect_second[i], boundRect_second[j])<dis_y  && dist_x(boundRect_second[i], boundRect_second[j])<dis_y_x))
                {//x方向或y方向上可以合并
                    rectangle( drawing, min_tl(boundRect_second[i],boundRect_second[j]), max_br(boundRect_second[i],boundRect_second[j]), color,  CV_FILLED, 8, 0 );
                    
                }
            
            }
        }
        boundRect_second.clear();
        vector <Rect>().swap(boundRect_second);
        contours.clear();
        hierarchy.clear();
    }
}
int CutRow(IplImage *result,IplImage * TransImage,int row)
{
    
    IplImage * TmpImage = cvCloneImage(TransImage);
    cvSaveImage("/Users/richard/Desktop/one.jpg", TmpImage);
    
    CvMemStorage *OutlineSto=cvCreateMemStorage();
    CvSeq *Outlineseq=NULL;
    
    int Num=cvFindContours(TmpImage,OutlineSto,&Outlineseq,sizeof(CvContour),CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE,cvPoint(0,0));
    int x = 0;
    int* Rect_x = new int[Num];
    
    for (CvSeq *c=Outlineseq;c!=NULL;c=c->h_next)
    {
        
        CvRect RectBound=cvBoundingRect(c,0);
        if(RectBound.height > 1 && (RectBound.width * 1.0 / RectBound.height) > 0.2)
        {
            Rect_x[x]=RectBound.x;
            x++;
        }
    }
    /*数字分割*/
    qsort(Rect_x,Num,sizeof(int),cmp);
    //分割数字并单独存储
    for(CvSeq *c=Outlineseq;c!=NULL;c=c->h_next)
    {
        CvRect RectBound=cvBoundingRect(c,0);
        //     CvRect CutRect=cvRect(RectBound.x,RectBound.y,RectBound.width,RectBound.height);
        if(RectBound.height > 1 && (RectBound.width * 1.0 / RectBound.height) > 0.2)
        {
            CvRect CutRect=cvRect(RectBound.x,RectBound.y,RectBound.width,RectBound.height);
            
            IplImage *ImgNo=cvCreateImage(cvSize(CutRect.width,CutRect.height),8,1);
            
            cvSet(ImgNo,cvScalar(0),0); //将图像填充为黑色
            
            cvSetImageROI(result,CutRect);
            cvCopy(result,ImgNo);
            cvResetImageROI(result);
            int col=0;
            for(int i=0;i<Num;i++)
            {
                if(Rect_x[i]==RectBound.x)
                {
                    col=i;
                    
                    break;
                }
            }
            //为图像存储路径分配内存
            char *SavePath=(char *)malloc(80*sizeof(char));
            if (SavePath==NULL)
            {
                printf("分配内存失败");
                exit(1);
            }
            sprintf(SavePath,"/Users/richard/Desktop/毕设/截取result/Num%d(%d).jpg",row,col+1);
            cvSaveImage(SavePath,ImgNo);
            free(SavePath);
            SavePath=NULL;
            cvReleaseImage(&ImgNo);
        }
    }
    delete Rect_x;
    cvReleaseMemStorage(&OutlineSto);
    cvReleaseImage(&TmpImage);
    return Num;
}

int CutNum(IplImage *RotateRow, cv::Mat& src, char* filename)
{

   cv::Mat mat = cv::cvarrToMat(RotateRow) ;
    vector<vector<Point> > contours;//存储轮廓的点集合
    vector<Vec4i> hierarchy;//构建轮廓的层次结构

    Mat dst;
    mat.copyTo(dst);
    adaptiveThreshold(dst, dst, 255, 0, cv::THRESH_BINARY_INV, 5, 10);
    adaptiveThreshold(mat, mat, 255, 0, cv::THRESH_BINARY_INV, 5, 10);
    
    Scalar color = Scalar( 255,255,255 );
    Scalar color_b = Scalar( 0,0,0 );
    Scalar color_r = Scalar( 0,0,255 );
    
    Mat element(5,5,CV_8U,Scalar(1));

    //计算轮廓凸包
    findContours( mat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Rect> boundRect( contours.size() );
    vector<vector<Point> >hull( contours.size() );
   
    for( int i = 0; i < contours.size(); i++ )
    {
        convexHull( Mat(contours[i]), hull[i], false );//凸包计算
    }
    // 用凸包重绘连通域
    Mat drawing = Mat::zeros( mat.size(), CV_8UC1 );
    
    
    for( int i = 0; i< contours.size(); i++ )
    {
        boundRect[i] = boundingRect( Mat(contours[i]) );
        double area = contourArea(contours[i]);
        double dx = density(boundRect[i].width,boundRect[i].height,area);
        double hwRate = HWRate(boundRect[i].width, boundRect[i].height);
        int pix_n ;
        
        if (area>minarea && dx>0.06 && hwRate>0.06 )
        {
            pix_n =  pix_num(dst, boundRect[i]);
            if (pix_n>40) {
                drawContours( drawing, hull, i, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
            }

        }
    }
    boundRect.clear();
    vector <Rect>().swap(boundRect);
    hull.clear();
    contours.clear();
    hierarchy.clear();
     imwrite("/Users/richard/Desktop/1.jpg", drawing );
 
    //合并连通域（字间）
    /*
     2次;
     dis_x<10 && dis_y<2;
     dis_y<2 && dis_x<2;
    */
    connect_words(drawing,contours,hierarchy,1,30,12,5,5);
    imwrite("/Users/richard/Desktop/2.jpg", drawing );
    //合并连通域（行间）
    /*
     1次;
     dis_x<25 && dis_y<20;
     dis_y<50 && dis_x<15;
     */

    connect_words(drawing,contours,hierarchy,1,10,50,15,20);
    imwrite("/Users/richard/Desktop/3.jpg", drawing );
 
    //合并连通域（行间2次）
    /*
     1次;
     dis_x<10 && dis_y<10;
     dis_y<30 && dis_x<15;
     */
    
    connect_words(drawing,contours,hierarchy,1,15,25,15,8);
    imwrite("/Users/richard/Desktop/4.jpg", drawing );

     //合并相交连通域
    findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Rect> boundRect_3( contours.size() );
    for(int i = 0; i< contours.size(); i++)
    {
        boundRect_3[i] = boundingRect( Mat(contours[i]) );
    }
    for(int i = 0;i< contours.size();i++)
    {
        for(int j =0;j<contours.size();j++)
        {
            if (rectA_intersect_rectB(boundRect_3[i], boundRect_3[j]))
            {
                rectangle(drawing, boundRect_3[i], color, CV_FILLED, 8, 0);
                rectangle(drawing, boundRect_3[j], color, CV_FILLED, 8, 0);
            }

        }
    }
    boundRect_3.clear();
    vector <Rect>().swap(boundRect_3);
    contours.clear();
    hierarchy.clear();
    
   //后处理，去掉面积小的连通域
    findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Rect> boundRect_4( contours.size() );
    
    for(int i = 0; i< contours.size(); i++)
    {

        boundRect_4[i] = boundingRect( Mat(contours[i]) );
        double area = boundRect_4[i].width*boundRect_4[i].height;
        if (area<500)
        {
            rectangle(drawing, boundRect_4[i], color_b, CV_FILLED, 8, 0);
        
        }
    }
    boundRect_4.clear();
    vector <Rect>().swap(boundRect_4);
    contours.clear();
    hierarchy.clear();

    imwrite("/Users/richard/Desktop/5.jpg", drawing );

    
    CvMemStorage *OutlineSto=cvCreateMemStorage();
    CvSeq *Outlineseq=NULL;

    
    findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Rect> boundRect_third( contours.size() );
    int Num = contours.size();

    int x = 0;
    int* Rect_x = new int[Num];
    int* Rect_y = new int[Num];

    for (int i=0; i<Num; i++)
    {
        boundRect_third[i] = boundingRect( Mat(contours[i]) );
    }

    for (int i= 0; i<Num; i++)
    {
            cv::rectangle(src, boundRect_third[i], color_r);//画外接矩形
            Rect_x[x] = boundRect_third[i].x;
            Rect_y[x] = boundRect_third[i].y;
            x++;
    }
    
    IplImage * result = cvLoadImage(filename,0);
    imwrite("/Users/richard/Desktop/label.jpg", src);
 
    
/*版块分割*/
    qsort(Rect_x,Num,sizeof(int),cmp);
    qsort(Rect_y,Num,sizeof(int),cmp);
 //分割版块并单独存储
    int col=0;
    int row=0;
    for (int i= 0; i<Num; i++)
    {
            CvRect CutRect=cvRect(boundRect_third[i].x,boundRect_third[i].y,
                                  boundRect_third[i].width,boundRect_third[i].height);
            IplImage *ImgNo=cvCreateImage(cvSize(CutRect.width,CutRect.height),8,1);
            
            cvSet(ImgNo,cvScalar(0),0); //将图像填充为黑色
            
            cvSetImageROI(result,CutRect);
            cvCopy(result,ImgNo);
            cvResetImageROI(result);
            for(int j=0;j<Num;j++)
            {
                if(Rect_x[j]==boundRect_third[i].x)
                {
                    col=j;
                    break;
                }
            }
            for(int j=0;j<Num;j++)
            {
                if(Rect_y[j]==boundRect_third[i].y)
                {
                    row=j;
                    break;
                }
            }
            //为图像存储路径分配内存
            char *SavePath=(char *)malloc(80*sizeof(char));
            if (SavePath==NULL)
            {
                printf("分配内存失败");
                exit(1);
            }
            sprintf(SavePath,"/Users/richard/Desktop/截取result/Num%d(%d).jpg",row+1,col+1);
            cvSaveImage(SavePath,ImgNo);
            free(SavePath);
            SavePath=NULL;
            cvReleaseImage(&ImgNo);
    }
    delete Rect_x;
    boundRect_third.clear();
    vector <Rect>().swap(boundRect_third);
    contours.clear();
    hierarchy.clear();
    return Num;

}

/*
int CutAndDisaplay(char* pagename,char* filename_label )
{
    
    IplImage * result = cvLoadImage(filename_label,0);
    IplImage * TmpImage = cvLoadImage(pagename,0);
    Mat src = imread(filename_label);
    CutNum(TmpImage,src,filename_label,1);
    imwrite("/Users/richard/Desktop/label.jpg",src);
    
    int Num = CutRow(result,TmpImage,0);
    for(int i=0;i < Num;i++)
    {
        result = cvLoadImage("/Users/richard/Desktop/毕设/截取result/Num0(%d).jpg",i);
        if(result == NULL)
            continue;
        else
        {
            TmpImage = cvLoadImage(filename,0);
            CutRow(result,TmpImage,i);
            
        }
    }
}
*/
