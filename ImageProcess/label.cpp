//
//  label.cpp
//  ImageProcess
//
//  Created by richard on 1/12/17.
//  Copyright © 2017 richard. All rights reserved.
//

#include "label.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <iostream>
using namespace cv;
using namespace std;

void getConnectedDomain(Mat& src, vector<Rect>& boundingbox)//boundingbox为最终结果，存放各个连通域的包围盒
{
    int img_row = src.rows;
    int img_col = src.cols;
    Mat flag = Mat::zeros(Size(img_col, img_row), CV_8UC1);//标志矩阵，为0则当前像素点未访问过
    for (int i = 0; i < img_row; i++)
    {
        for (int j = 0; j < img_col; j++)
        {
            if (src.ptr<uchar>(i)[j] == 0 && flag.ptr<uchar>(i)[j] == 0)
            {
                stack<Point2f> cd;
                cd.push(Point2f(j, i));
                flag.ptr<uchar>(i)[j] = 1;
                int minRow = i, minCol = j;//包围盒左、上边界
                int maxRow = i, maxCol = j;//包围盒右、下边界
                while (!cd.empty())
                {
                    Point2f tmp = cd.top();
                    if (minRow > tmp.y)//更新包围盒
                        minRow = tmp.y;
                    if (minCol > tmp.x)
                        minCol = tmp.x;
                    if (maxRow < tmp.y)
                        maxRow = tmp.y;
                    if (maxCol < tmp.x)
                        maxCol = tmp.x;
                    cd.pop();
                    Point2f p[4];//邻域像素点，这里用的四邻域
                    p[0] = Point2f(tmp.x - 1 > 0 ? tmp.x - 1 : 0, tmp.y);
                    p[1] = Point2f(tmp.x + 1 < img_col - 1 ? tmp.x + 1 : img_row - 1, tmp.y);
                    p[2] = Point2f(tmp.x, tmp.y - 1 > 0 ? tmp.y - 1 : 0);
                    p[3] = Point2f(tmp.x, tmp.y + 1 < img_row - 1 ? tmp.y + 1 : img_row - 1);
                    for (int m = 0; m < 4; m++)
                    {
                        int x = p[m].y;
                        int y = p[m].x;
                        if (src.ptr<uchar>(x)[y] == 0 && flag.ptr<uchar>(x)[y] == 0)//如果未访问，则入栈，并标记访问过该点
                        {
                            cd.push(p[m]);
                            flag.ptr<uchar>(x)[y] = 1;
                        }
                    }
                }
                Rect rect(Point2f(minCol, minRow), Point2f(maxCol + 1, maxRow + 1));
                boundingbox.push_back(rect);
            }
        }
    }
}


int label (unsigned short *mask, IplImage *img1)
{
    unsigned char *pb1=(unsigned char*)img1->imageData;
    int height=img1->height;
    int width=img1->width;
    int k,j;
    
    for( k=0; k<width*height; k++)
        mask[k]=0;
    
    int cw=100;//初始区域数目
loop:
    bool *col=new bool[cw*cw];
    for(k=0; k<cw*cw; k++)
        col[k]=false;
    
    //第一次扫描
    unsigned short lab=1;
    for(k=1;k<height;k++)
        for( j=1;j<width-1;j++)
            if(pb1[k*width+j]==255)
                if((mask[k*width+j-1]+mask[(k-1)*width+j-1]+mask[(k-1)*width+j]+mask[(k-1)*width+j+1])==0)
                {
                    
                    mask[k*width+j]=lab;
                    lab=lab+1;
                    if(lab>cw)
                    {
                        
                        delete col;
                        cw+=200;
                        goto loop;
                        
                    }
                    
                }
                else
                {
                    
                    if(mask[k*width+j-1]!=0)
                        mask[k*width+j]=mask[k*width+j-1];
                    
                    if(mask[(k-1)*width+j-1]!=0)
                        if(mask[k*width+j]==0)
                            mask[k*width+j]=mask[(k-1)*width+j-1];
                        else
                        {
                            
                            col[mask[k*width+j]*cw+mask[(k-1)*width+j-1]]=true;
                            col[mask[(k-1)*width+j-1]*cw+mask[k*width+j]]=true;
                            
                        }
                    
                    if(mask[(k-1)*width+j]!=0)
                        if(mask[k*width+j]==0)
                            mask[k*width+j]=mask[(k-1)*width+j];
                        else
                        {
                            
                            col[mask[k*width+j]*cw+mask[(k-1)*width+j]]=true;
                            col[mask[(k-1)*width+j]*cw+mask[k*width+j]]=true;
                            
                        }
                    
                    if(mask[(k-1)*width+j+1]!=0)
                        if(mask[k*width+j]==0)
                            mask[k*width+j]=mask[(k-1)*width+j+1];
                        else
                        {
                            
                            col[mask[k*width+j]*cw+mask[(k-1)*width+j+1]]=true;
                            col[mask[(k-1)*width+j+1]*cw+mask[k*width+j]]=true;
                            
                        }
                    
                }
    
    if(lab==1)
        return 0;
    
    //等价关系合并
    bool *col2=new bool[lab*lab];
    for(k=0; k<lab*lab; k++)
        col2[k]=false;
    for(k=1;k<lab;k++)
        for(j=1;j<lab;j++)
            if(col[k*cw+j]==true||k==j)
                col2[k*lab+j]=true;
    delete[] col;
    
    for(k=1;k<lab;k++)
        for(j=1;j<lab;j++)
            if(col2[j*lab+k]==true)
                for(int i=1;i<lab;i++)
                    col2[j*lab+i]=col2[j*lab+i]||col2[k*lab+i];
    
    
    int *col3=new int [lab];
    int *col3_s=new int[lab];
    for(k=1;k<lab;k++)
    {
        
        int c1, c2;
        for(j=1;j<lab;j++)
            if(col2[k*lab+j]==true)
            {
                
                c1=j;
                break;
                
            }
        for(j=1;j<lab;j++)
            if(col2[j*lab+k]==true)
            {
                
                c2=j;
                break;
                
            }
        col3[k]=min(c1,c2);
        col3_s[k]=col3[k];
        
    }
    
    
    for(k=1;k<lab-1;k++)
    {
        
        int t;
        for(j=1;j<lab-k;j++)
            if(col3_s[j]>col3_s[j+1])
            {
                
                t=col3_s[j];
                col3_s[j]=col3_s[j+1];
                col3_s[j+1]=t;
                
            }
        
    }
    
    int *ind=new int[lab];
    ind[col3_s[1]]=1;
    int c = 2;
    for(k=1;k<lab-1;k++)
        if(col3_s[k+1]!=col3_s[k])
        {
            
            ind[col3_s[k+1]]=c;
            c++;
            
        }
    
    //第2次扫描
    for(k=1;k<height;k++)
        for( j=1;j<width-1;j++)
            if(mask[k*width+j]!=0)
                mask[k*width+j]=ind[col3[mask[k*width+j]]];
    
    delete[] col2;
    delete[] col3;
    delete[] col3_s;
    delete[] ind;
    
    return(c-1);//返回区域数目
    
}
/*
void  yuBwLabel( const Mat &bwImg, Mat &labImg )
{
    assert( bwImg.type()==CV_8UC1 );
    labImg.create( bwImg.size(), CV_32SC1 ); // bwImg.convertTo( labImg, CV_32SC1 );
    labImg = Scalar(0);
    labImg.setTo( Scalar(1), bwImg );
    assert( labImg.isContinuous() );
    const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
    int label = 1;
    vector<int> labelSet;
    labelSet.push_back(0);
    labelSet.push_back(1);
    // the first pass
    int *data_prev = (int*)labImg.data; // 0-th row : int* data_prev = labImg.ptr<int>(i-1);
    int *data_cur = (int*)( labImg.data + labImg.step ); // 1-st row : int* data_cur = labImg.ptr<int>(i);
    for( int i=1; i<Rows; i++ ){
        data_cur++;
        data_prev++;
        for( int j=1; j<Cols; j++, data_cur++, data_prev++ ){
            if( *data_cur!=1 )
                continue;
            int left = *(data_cur-1);
            int up = *data_prev;
            int neighborLabels[2];
            int cnt = 0;
            if( left>1 )
                neighborLabels[cnt++] = left;
            if( up>1 )
                neighborLabels[cnt++] = up;
            if( !cnt ){
                labelSet.push_back( ++label );
                labelSet[label] = label;
                *data_cur = label;
                continue;
            }
            int smallestLabel = neighborLabels[0];
            if( cnt==2 && neighborLabels[1]<smallestLabel )
                smallestLabel = neighborLabels[1];
            *data_cur = smallestLabel;
            // 保存最小等价表
            for( int k=0; k<cnt; k++ ){
                int tempLabel = neighborLabels[k];
                int& oldSmallestLabel = labelSet[tempLabel];
                if( oldSmallestLabel > smallestLabel ){
                    labelSet[oldSmallestLabel] = smallestLabel;
                    oldSmallestLabel = smallestLabel;
                }
                else if( oldSmallestLabel<smallestLabel )
                    labelSet[smallestLabel] = oldSmallestLabel;
            }
        }
        data_cur++;
        data_prev++;
    }
    // 更新等价对列表,将最小标号给重复区域
    for( size_t i = 2; i < labelSet.size(); i++ ){
        int curLabel = labelSet[i];
        int preLabel = labelSet[curLabel];
        while( preLabel!=curLabel ){
            curLabel = preLabel;
            preLabel = labelSet[preLabel];
        }
        labelSet[i] = curLabel;
    }
    // second pass
    data_cur = (int*)labImg.data;
    for( int i=0; i<Rows; i++ ){
        for( int j=0; j<bwImg.cols-1; j++, data_cur++ )
            *data_cur = labelSet[ *data_cur ];
        data_cur++;
    }
}
*/
/*
void  yuBwLabel( const Mat &_binImg, Mat & _lableImg )
{
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows-1; i++)
    {
        int* data= _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols-1; j++)
        {
            if (data[j] == 1)
            {
                std::stack<std::pair<int,int>> neighborPixels ;
                neighborPixels.push(std::pair<int,int>(i,j)) ;     // pixel position: <i,j>
                ++label ;  // begin with a new label
                while (!neighborPixels.empty())
                {
                    // get the top pixel on the stack and label it with the same label
                    std::pair<int,int> curPixel = neighborPixels.top() ;
                    int curX = curPixel.first ;
                    int curY = curPixel.second ;
                    _lableImg.at<int>(curX, curY) = label ;
                    
                    // pop the top pixel
                    neighborPixels.pop() ;
                    
                    // push the 4-neighbors (foreground pixels)
                    if (curY != 1 && _lableImg.at<int>(curX, curY-1) == 1) //左值
                        neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;
                    if ( curY != rows && _lableImg.at<int>(curX, curY+1) == 1)// 右值
                        neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;
                    if (curX != 0 && _lableImg.at<int>(curX-1, curY) == 1)// 上值
                        neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
                    if (curX != cols && _lableImg.at<int>(curX+1, curY) == 1)//下值 
                        neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;
                }
            }  
        }  
    }
}
*/
/*Seed-Filling种子填充方法*/
/*
void  yuBwLabel( const Mat &_binImg, Mat &_lableImg )
{
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
 
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
 
    int label = 1 ;  // start by 2
 
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows-1; i++)
    {
        int* data= _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols-1; j++)
        {
            if (data[j] == 1)
            {
                std::stack<std:: pair<int,int>> neighborPixels ;
                neighborPixels.push(std::pair<int,int>(i,j)) ;     // pixel position: <i,j>
                ++label ;  // begin with a new label
                while (!neighborPixels.empty())
                {
                    // get the top pixel on the stack and label it with the same label
                    std::pair<int,int> curPixel = neighborPixels.top() ;
                    int curX = curPixel.first ;
                    int curY = curPixel.second ;
                    _lableImg.at<int>(curX, curY) = label ;
                    
                    // pop the top pixel
                    neighborPixels.pop() ;
                    
                    // push the 4-neighbors (foreground pixels)
                    if (_lableImg.at<int>(curX, curY-1) == 1)
                    {// left pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;
                    }
                    if (_lableImg.at<int>(curX, curY+1) == 1)
                    {// right pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;
                    }
                    if (_lableImg.at<int>(curX-1, curY) == 1)
                    {// up pixel
                        neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
                    }
                    if (_lableImg.at<int>(curX+1, curY) == 1)
                    {// down pixel
                        neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;
                    }
                }
            }  
        }  
    }
}*/
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
    // connected component analysis (4-component)
    // use two-pass algorithm
    // 1. first pass: label each foreground pixel with a label
    // 2. second pass: visit each labeled pixel and merge neighbor labels
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0
    
    
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
    
    // 1. first pass
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    std::vector<int> labelSet ;
    labelSet.push_back(0) ;   // background: 0
    labelSet.push_back(1) ;   // foreground: 1
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows; i++)
    {
        int* data_preRow = _lableImg.ptr<int>(i-1) ;
        int* data_curRow = _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols; j++)
        {
            if (data_curRow[j] == 1)
            {
                std::vector<int> neighborLabels ;
                neighborLabels.reserve(2) ;
                int leftPixel = data_curRow[j-1] ;
                int upPixel = data_preRow[j] ;
                if ( leftPixel > 1)
                {
                    neighborLabels.push_back(leftPixel) ;
                }
                if (upPixel > 1)
                {
                    neighborLabels.push_back(upPixel) ;
                }
                
                if (neighborLabels.empty())
                {
                    labelSet.push_back(++label) ;  // assign to a new label
                    data_curRow[j] = label ;
                    labelSet[label] = label ;
                }
                else
                {
                    std::sort(neighborLabels.begin(), neighborLabels.end()) ;
                    int smallestLabel = neighborLabels[0] ;
                    data_curRow[j] = smallestLabel ;
                    
                    // save equivalence
                    for (size_t k = 1; k < neighborLabels.size(); k++)
                    {
                        int tempLabel = neighborLabels[k] ;
                        int& oldSmallestLabel = labelSet[tempLabel] ;
                        if (oldSmallestLabel > smallestLabel)
                        {
                            labelSet[oldSmallestLabel] = smallestLabel ;
                            oldSmallestLabel = smallestLabel ;
                        }
                        else if (oldSmallestLabel < smallestLabel)
                        {
                            labelSet[smallestLabel] = oldSmallestLabel ;
                        }
                    }
                }
            }
        }
    }
    
    // update equivalent labels
    // assigned with the smallest label in each equivalent label set
    for (size_t i = 2; i < labelSet.size(); i++)
    {
        int curLabel = labelSet[i] ;
        int preLabel = labelSet[curLabel] ;
        while (preLabel != curLabel)
        {
            curLabel = preLabel ;
            preLabel = labelSet[preLabel] ;
        }
        labelSet[i] = curLabel ;
    }
    
    
    // 2. second pass
    for (int i = 0; i < rows; i++)
    {
        int* data = _lableImg.ptr<int>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int& pixelLabel = data[j] ;
            pixelLabel = labelSet[pixelLabel] ;
        }  
    }  
}
void icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
    // connected component analysis (4-component)
    // use seed filling algorithm
    // 1. begin with a foreground pixel and push its foreground neighbors into a stack;
    // 2. pop the top pixel on the stack and label it with the same label until the stack is empty
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0
    
    
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows-1; i++)
    {
        int* data= _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols-1; j++)
        {
            if (data[j] == 1)
            {
                std::stack<std::pair<int,int>> neighborPixels ;
                neighborPixels.push(std::pair<int,int>(i,j)) ;     // pixel position: <i,j>
                ++label ;  // begin with a new label
                while (!neighborPixels.empty())
                {
                    // get the top pixel on the stack and label it with the same label
                    std::pair<int,int> curPixel = neighborPixels.top() ;
                    int curX = curPixel.first ;
                    int curY = curPixel.second ;
                    _lableImg.at<int>(curX, curY) = label ;
                    
                    // pop the top pixel
                    neighborPixels.pop() ;
                    
                    // push the 4-neighbors (foreground pixels)
                    if (_lableImg.at<int>(curX, curY-1) == 1)
                    {// left pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;
                    }
                    if (_lableImg.at<int>(curX, curY+1) == 1)
                    {// right pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;
                    }
                    if (_lableImg.at<int>(curX-1, curY) == 1)
                    {// up pixel
                        neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
                    }
                    if (_lableImg.at<int>(curX+1, curY) == 1)
                    {// down pixel
                        neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;
                    }
                }
            }  
        }  
    }  
}
