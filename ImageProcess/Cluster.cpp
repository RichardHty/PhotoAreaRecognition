//
//  Cluster.cpp
//  ImageProcess
//
//  Created by richard on 1/15/17.
//  Copyright © 2017 richard. All rights reserved.
//

#include "Cluster.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
IplImage* cluster(IplImage* img)
{
    int i,j;
    CvMat *samples=cvCreateMat((img->width)*(img->height),1,CV_32FC3);//创建样本矩阵，CV_32FC3代表32位浮点3通道（彩色图像）
    CvMat *clusters=cvCreateMat((img->width)*(img->height),1,CV_32SC1);//创建类别标记矩阵，CV_32SF1代表32位整型1通道
    int k=0;
    for (i=0;i<img->width;i++)
    {
        for (j=0;j<img->height;j++)
        {
            CvScalar s;
        //获取图像各个像素点的三通道值（RGB）
            s.val[0]=(float)cvGet2D(img,j,i).val[0];
            s.val[1]=(float)cvGet2D(img,j,i).val[1];
            s.val[2]=(float)cvGet2D(img,j,i).val[2];
            cvSet2D(samples,k++,0,s);//将像素点三通道的值按顺序排入样本矩阵
        }
    }
    int nCuster=2;//聚类类别数，自己修改。
    cvKMeans2(samples,nCuster,clusters,cvTermCriteria(CV_TERMCRIT_ITER,100,1.0));//开始聚类，迭代100次，终止误差1.0

    IplImage *bin=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);//创建用于显示的图像，二值图像
    k=0;
    int val=0;
    float step=255/(nCuster-1);
    for (i=0;i<img->width;i++)
    {
        for (j=0;j<img->height;j++)
        {
            val=(int)clusters->data.i[k++];
            CvScalar s;
            s.val[0]=val*step;//这个是将不同类别取不同的像素值，
            cvSet2D(bin,j,i,s); //将每个像素点赋值
        }
    }

    
 //   cvSaveImage("/Users/richard/Desktop/clustered.jpg", bin);
    cvReleaseImage( &img ); //释放图像
    return bin;
}
