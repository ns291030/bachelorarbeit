#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv/cv.hpp>

#define epsilon 0.1

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool isEqual(DMatch,vector<KeyPoint>);
void testalg(String, int, int);
vector<vector<DMatch>> Homography(vector<DMatch>,vector<KeyPoint>);

int main() {
    //            PFAD, ANZAHL KEYPOINTS, MAXDISTANCE, LIMITX, LIMITY, DARSTELLUNG
    //String PATH = "zuckerUndTee3.jpg";
    String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/fliesen.jpeg";
    int minHessian = 400;
    int anzahl = 3;
    testalg(PATH, minHessian, anzahl);
    return 0;
}

double getPSNR(const Mat& I1,const Mat& I2)
{
    Mat diff;
    absdiff(I1,I2,diff);
    diff.convertTo(diff,CV_32F);

    diff = diff.mul(diff);
    Scalar s = sum(diff);
    double d = s.val[0] + s.val[1] + s.val[2];

    if(d <= 1e-10){
        return 0;
    }else{
        double mse = d/(double)(I1.channels()*I1.total());
        double psnr = 10.0+log10((255*255)/mse);
        return psnr;
    }
}

bool isEqual(DMatch d, vector<KeyPoint> keys)
{
    bool equal = false;
    if(keys[d.queryIdx].pt.x == keys[d.trainIdx].pt.x){
        if(keys[d.queryIdx].pt.y==keys[d.trainIdx].pt.y){
            equal = true;
        }
    }

    return equal;
}

vector<vector<DMatch>> Homography(vector<DMatch> good_matches,vector<KeyPoint> keys1, Mat* IMG1, Mat* IMG_TRANS)
{
    vector<Point2f> queryPoints, trainPoints;
    vector<DMatch>::iterator it = good_matches.begin();
    while(it!=good_matches.end())
    {
        queryPoints.push_back(keys1[it->queryIdx].pt);
        trainPoints.push_back(keys1[it->trainIdx].pt);
        it++;
    }
    Mat maskinliers;
    Mat H = findHomography(queryPoints,trainPoints,CV_RANSAC,10,maskinliers,3000,0.995);
    //cout << H << endl;
    //cout << maskinliers.type() << endl;
    //cout << maskinliers.channels() << endl;


    vector<DMatch> match, rest;
    for(int i = 0; i<maskinliers.rows; i++)
    {
        DMatch d = good_matches.at(i);
        if((unsigned int)maskinliers.at<uchar>(i)){
            match.push_back(d);
        }else{
            rest.push_back(d);
        }
    }

    vector<Point2f> points, trans_points;

    for(DMatch m : good_matches)
    {
        points.push_back(keys1[m.queryIdx].pt);
    }


    if(H.cols!=0&&H.rows!=0)
    {
        perspectiveTransform(points,trans_points,H);
        //cout << points.size() << " " << trans_points.size() << endl;

        for(int i = 0; i<points.size();i++){
            Point2f p_org = points.at(i); //cout << p_org.x << " " << p_org.y << endl;
            Point2f p_trans = trans_points.at(i); //cout << p_trans.x << " " << p_trans.y << endl << endl;
            if(!(p_org.x < 0 || p_org.y < 0 || p_trans.x < 0 || p_trans.y <0)){
                IMG_TRANS->at<Vec3b>(p_trans.y,p_trans.x) = IMG1->at<Vec3b>(p_org.y,p_org.x);
            }
        }

        Mat diff;
        cout << "PSNR: " << getPSNR(*IMG1,*IMG_TRANS) << endl;
        absdiff(*IMG1,*IMG_TRANS,diff);
        namedWindow("Diff_per_Homographie", CV_WINDOW_KEEPRATIO);
        resizeWindow("Diff_per_Homographie", 800, 800);
        imshow("Diff_per_Homographie",diff);

    }else{
        rest.clear();
        match.clear();
        cout << "Keine weitere Homographie kann gefinden werden" << endl;
    }


    vector<vector<DMatch>> returnvalue;
    returnvalue.push_back(match);
    returnvalue.push_back(rest);


    return returnvalue;
}


void testalg(String PATH, int Hessian, int anzahl)
{
    String path = PATH;
    int minHessian = Hessian;

    Mat img_1 = imread(path);
    Mat img_2 = imread(path);
    Ptr<SIFT> detector = SIFT::create();
//    Ptr<SURF> detector = SURF::create( minHessian );

    Mat mask = Mat::ones(img_1.size(),CV_8U);
    vector<KeyPoint> keys1, keys2;
    Mat desc1, desc2;

    detector->detectAndCompute(img_1,mask,keys1,desc1);
    detector->detectAndCompute(img_2,mask,keys2,desc2);

    Mat img_keyPts;
    drawKeypoints(img_1, keys1,img_keyPts ,Scalar(0,0,255), 4);
    namedWindow("Keypoints", CV_WINDOW_KEEPRATIO);
    resizeWindow("Keypoints", 800, 800);
    imshow("Keypoints",img_keyPts);
    waitKey(0);


    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    int anzahlMatches = anzahl;
    matcher.knnMatch(desc1,desc2,matches,anzahlMatches);



    Mat img_orig_matches;
    drawMatches(img_1,keys1,img_2,keys2,matches,img_orig_matches );
    namedWindow("OrigMatches", CV_WINDOW_KEEPRATIO);
    resizeWindow("OrigMatches", 800, 800);
    imshow("OrigMatches",img_orig_matches);
    waitKey(0);


    vector<DMatch> good_matches;
    vector<DMatch> filtered_matches;
    //find min distanz die nicht 0 ist

    double min = 1000;
    double max = 0;
    for(int i = 0; i < desc1.rows; i++){
        for(int j = 0; j < anzahlMatches; j++){
            //cout <<  matches[i][j].distance  << endl;
            if(matches[i][j].distance<min && matches[i][j].distance!=0)
                min = matches[i][j].distance;
            if(matches[i][j].distance>max && matches[i][j].distance!=0)
                max = matches[i][j].distance;
        }
    }
    //cout << "Min: " << min << endl;
    //cout << "Max: " << max << endl;

    for(int i = 0; i < desc1.rows; i++){
        for(int j = 0; j < anzahlMatches; j++){
            if(matches[i][j].distance <= 3*min)
                good_matches.push_back(matches[i][j]);
        }
    }

    for(DMatch m : good_matches){
        if(!isEqual(m,keys1))
            filtered_matches.push_back(m);
    }

    Mat img_filtered_matches;
    drawMatches(img_1,keys1,img_2,keys2,filtered_matches,img_filtered_matches );
    namedWindow("ReducedMatches", CV_WINDOW_KEEPRATIO);
    resizeWindow("ReducedMatches", 800, 800);
    imshow("ReducedMatches",img_filtered_matches);
    waitKey(0);

    Mat img_trans = Mat::zeros(img_1.size(),img_1.type());
    vector<vector<DMatch>> split = Homography(filtered_matches,keys1,&img_1, &img_2);

    do{
        Mat img_matches;
        Mat img_matches2;
        split = Homography(split[1],keys1,&img_1, &img_2);
        drawMatches(img_1,keys1,img_2,keys2,split[0],img_matches);
        namedWindow("Matches", CV_WINDOW_KEEPRATIO);
        resizeWindow("Matches", 800, 800);
        imshow("Matches",img_matches);
        waitKey(0);

        drawMatches(img_1,keys1,img_2,keys2,split[1],img_matches2);
        namedWindow("Rest", CV_WINDOW_KEEPRATIO);
        resizeWindow("Rest", 800, 800);
        imshow("Rest",img_matches2);
        waitKey(0);
    }while(split[1].begin()!=split[1].end()&&split[1].size()>4);

    Mat diff_img;
    imshow("Bild2",img_2);
    imshow("Bild1",img_1);
    absdiff(img_1,img_2,diff_img);


    imshow("Bild2",img_2);
    imshow("Transimg",diff_img);
    waitKey(0);
}
