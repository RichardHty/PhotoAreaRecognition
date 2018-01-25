

//Hough变换在倾斜角度大时效果好，倾斜角度小时有过度校正的情况


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#include <label.hpp>
#include <ImgPre.hpp>
#include <ShowResult.hpp>
#include <PageNum.hpp>
#include <EdgeDetect.hpp>
#include <ProfilePro.hpp>
#include <Cluster.hpp>
#include <otsu.hpp>
#include <compensate.hpp>

using namespace cv;
using namespace std;


/*
 *** 连通域划分
 */
int main()
{
  
   Mat labImg,srcImg,colorLab;
    IplImage* PageImg = NULL;
    int i=0;
    char* filename = "/Users/richard/Desktop/教辅题_20.jpg";
    char* pagename = "/Users/richard/Desktop/vert.jpg";

    
 
    i = Imgpre(filename,srcImg); //-31.871/06、31.2719
    if ( i == -1)
        return 0;
    /*
    Imgpre_Hough(filename, srcImg);
    Imgpre_KHT(filename, srcImg);//-27.4615/06、35.5409
   */
   filename = "/Users/richard/Desktop/finImg.jpg";

    Mat src2 = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    Mat dst2 = imread(filename,CV_LOAD_IMAGE_GRAYSCALE) ;
    

    int a=0 ;
    int b = 0;
    project(src2,a,b);
    
    imgROI( src2,src2,labImg,0,6);

    //用拉普拉斯算子进行边缘锐化
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;
    
    cv::filter2D(labImg,labImg,labImg.depth(),kernel);
  
    imwrite("/Users/richard/Desktop/finImg.jpg", labImg);
   
     src2 = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
     dst2 = imread(filename,CV_LOAD_IMAGE_GRAYSCALE) ;
 /*
    int* y_limit=new int[src2.cols];
    RLSA_D_Y(src2,y_limit,0.1,1);
    verticalRLSA(src2, dst2, y_limit,1);
   */
    int* x_limit=new int[dst2.rows];
    RLSA_D_X(dst2,x_limit,0.1,1);
    horizontalRLSA(dst2, dst2, x_limit,1);
   
    
    //开操作
    erode(dst2, dst2, Mat(),Point(0,0),1);
    dilate(dst2, dst2, Mat(),Point(0,0),1);
    
    imwrite("/Users/richard/Desktop/vert.jpg", dst2);
    
    PageImg = cvLoadImage(pagename,0);

    Mat src = imread(filename);

    cout<<"共找到"<<CutNum(PageImg,src,filename)<<"个版块"<<endl;

    return 0;
}




