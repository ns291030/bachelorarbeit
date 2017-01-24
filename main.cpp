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
void testalg(String, int, int, bool);
vector<vector<DMatch>> Homography(vector<DMatch>,vector<KeyPoint>);


int main() {
    //            PFAD, ANZAHL KEYPOINTS, MAXDISTANCE, LIMITX, LIMITY, DARSTELLUNG
    //String PATH = "zuckerUndTee3.jpg";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/board3.jpg";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/BigShips.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/City.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/GT_Fly.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/Kimono.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/ParkScene.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/Poznan_Street.png";
    //String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/testframes/Traffic.png";

    String PATH = "/home/nikolaj/Bilder/bachelorarbeittest/newTestset/IMG_9812.JPG";
    int minHessian = 400;
    int anzahl = 3;
    testalg(PATH, minHessian, anzahl, true);
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

    for(DMatch m : match)
    {
        points.push_back(keys1[m.queryIdx].pt);
    }


    if(H.cols!=0&&H.rows!=0)
    {
        cout << "Matches: " << match.size() << endl << "Rest: " << rest.size() << endl;
        //Möglichkeit 1: Bei den Karten ok Ergebnis, bei dem Schiff zum Beispiel kein Ergebnis
        Mat warped;
        warpPerspective(*IMG1,warped,H,IMG1->size());
        Mat diff;
        absdiff(warped,*IMG1,diff);


        //Möglichkeit 2:

        vector<Point2f> hull;
        //Mat hull_img=IMG1->clone();

        //
        // Mat mask_image(IMG1->size(),CV_8U,Scalar(0));
        convexHull(points,hull);
        vector<Point2f> hull_transformed;
        perspectiveTransform(hull,hull_transformed,H);

        cout << hull_transformed.size() << endl;

        //fillConvexPoly(mask_image,hull,Scalar(255));

        //Mat test;
        //test.copyTo(*IMG1,mask_image);


        float min_x = 10000, min_y = 10000, max_x = 0, max_y = 0;

        for(int i = 0; i<hull_transformed.size();i++)
        {
            if(hull_transformed[i].x<min_x)
                min_x=hull_transformed[i].x;
            if(hull_transformed[i].x>max_x)
                max_x=hull_transformed[i].x;
            if(hull_transformed[i].y<min_y)
                min_y=hull_transformed[i].y;
            if(hull_transformed[i].y>max_y)
                max_y=hull_transformed[i].y;
        }

        Rect region_of_interest = Rect(min_x,min_y,(max_x-min_x),(max_y-min_y));
        Mat img_roi_warped = warped(region_of_interest);
        Mat img_roi_orig = (*IMG1)(region_of_interest);

        namedWindow("Roi", CV_WINDOW_KEEPRATIO);
        resizeWindow("Roi", 800, 800);
        imshow("Roi",img_roi_warped);

        namedWindow("Differenz", CV_WINDOW_KEEPRATIO);
        resizeWindow("Differenz", 800, 800);
        imshow("Differenz",diff);

        cout << "PSNR: " << getPSNR(img_roi_warped,img_roi_orig) << endl;

        //namedWindow("Hull", CV_WINDOW_KEEPRATIO);
        //resizeWindow("Hull", 800, 800);
        //imshow("Hull",hull_img);


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


void testalg(String PATH, int Hessian, int anzahl, bool b)
{
    String path = PATH;
    int minHessian = Hessian;

    Mat img_1 = imread(path);
    Mat img_2 = imread(path);

    if(b){
        const float scale = 0.6;
        resize(img_1,img_1,cv::Size(0,0),scale,scale);
        resize(img_2,img_2,cv::Size(0,0),scale,scale);

        cout << "IMG gescaled" << endl;

    }

    Ptr<SIFT> detector = SIFT::create();

    Mat mask = Mat::ones(img_1.size(),CV_8U);
    vector<KeyPoint> keys1, keys2;
    Mat desc1, desc2;

    detector->detectAndCompute(img_1,mask,keys1,desc1);
    detector->detectAndCompute(img_2,mask,keys2,desc2);

    cout << "Keypoints detected" << endl;


    //Mat img_keyPts;
    //drawKeypoints(img_1, keys1,img_keyPts ,Scalar(0,0,255), 4);
    //namedWindow("Keypoints", CV_WINDOW_KEEPRATIO);
    //resizeWindow("Keypoints", 800, 800);
    //imshow("Keypoints",img_keyPts);
    //waitKey(0);


    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    int anzahlMatches = anzahl;
    matcher.knnMatch(desc1,desc2,matches,anzahlMatches);
    vector<DMatch> good_matches;
    vector<DMatch> filtered_matches;

    cout << "Matches gefunden" << endl;


    double min = 1000;
    double max = 0;
    /*for(int i = 0; i < desc1.rows; i++){
        for(int j = 0; j < anzahlMatches; j++){
            //cout <<  matches[i][j].distance  << endl;
            if(matches[i][j].distance<min && matches[i][j].distance!=0)
                min = matches[i][j].distance;
            if(matches[i][j].distance>max && matches[i][j].distance!=0)
                max = matches[i][j].distance;
        }
    }
    for(int i = 0; i < desc1.rows; i++){
        for(int j = 0; j < anzahlMatches; j++){
            if(matches[i][j].distance <= 3*min)
                good_matches.push_back(matches[i][j]);
        }
    }*/

    /*for(DMatch m : good_matches){
        if(!isEqual(m,keys1))
            filtered_matches.push_back(m);
    }*/

    for(vector<DMatch> v : matches){
        for(DMatch m : v){
            if(!isEqual(m,keys1))
            filtered_matches.push_back(m);
        }
    }

    cout << "Matches gefiltered" << endl;

    Mat img_trans = Mat::zeros(img_1.size(),img_1.type());
    vector<vector<DMatch>> split;
    vector<DMatch> empty;
    split.push_back(empty);
    split.push_back(filtered_matches);


    cout << "Homographien" << endl;

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

    waitKey(0);
}


