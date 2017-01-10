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

bool is_near(Point2f, Point2f, int, int);
vector<vector<DMatch>> split_objects(vector<DMatch>, vector<KeyPoint>, int, int, int);
Mat homography(vector<DMatch>, vector<DMatch>, vector<KeyPoint>);
double area(Point2f, Point2f, Point2f);
bool isEqual(DMatch, vector<KeyPoint>);
void algorithmus(String, int, int, int, int, int, int a = -1);
void testalg(String, int, int);
bool contains(vector<Point2f>, Point2f);
vector<vector<DMatch>> Homography(vector<DMatch>,vector<KeyPoint>);

int main()
{
    //            PFAD, ANZAHL KEYPOINTS, MAXDISTANCE, LIMITX, LIMITY, DARSTELLUNG
//    String PATH = "zuckerUndTee3.jpg";
    String PATH = "board3.jpg";
    int minHessian = 400; int anzahl = 3; int distance = 100; int limitx = 50; int limity = 50;
    testalg(PATH,minHessian,3);
    //algorithmus(PATH,minHessian,anzahl,distance,limitx,limity);
    return 0;
}

bool checkAreaQuotient(vector<Point2f> ref_points, vector<Point2f> obj_points, Point2f ref_p, Point2f obj_p)
{
    double ref_area_ref = area(ref_points[0],ref_points[1],ref_points[2]);
    double ref_area_obj = area(obj_points[0],obj_points[1],obj_points[2]);

    double newarea_ref = area(ref_points[0],ref_points[1],ref_p);
    double newarea_obj = area(obj_points[0],obj_points[1],obj_p);

    double quot1 = ref_area_ref/newarea_ref;
    double quot2 = ref_area_obj/newarea_obj;

    cout << abs(quot1-quot2) << endl;

    if(abs(quot1-quot2)<epsilon)
    {
        return true;
    }

    return false;
}

double area(Point2f a, Point2f b, Point2f c)
{
    double x11 = a.x-c.x;
    double x12 = a.y-c.y;
    double x21 = b.x-c.x;
    double x22 = b.y-c.y;

    double det = abs((x11*x22)-(x12*x21));
    return 0.5*det;
}

Mat homography(vector<DMatch> ref_obj ,vector<DMatch> obj1, vector<KeyPoint> keys_ref)
{
    vector<Point2f> a;
    vector<Point2f> b;

    for(DMatch m : ref_obj){
        a.push_back(keys_ref[m.queryIdx].pt);
        b.push_back(keys_ref[m.trainIdx].pt);
    }

    Mat hom = findHomography(a,b,RANSAC);

    return hom;
}

bool is_near(Point2f a, Point2f b, int limitx, int limity)
{

    return abs(a.x-b.x)<limitx && abs(a.y-b.y)<limity;
}

vector<vector<DMatch>> split_objects(vector<DMatch> ref_obj,vector<KeyPoint> keys,int limitx, int limity, int durchlauf)
{
    vector<vector<DMatch>> objects;

    vector<DMatch> obj;
    //vector<DMatch> ref_triangle;
    vector<Point2f> ref_triangle_points_a;
    vector<Point2f> ref_triangle_points_b;

    for(int i = 0; i<durchlauf; i++) {
        obj.clear();
        ref_triangle_points_a.clear();
        ref_triangle_points_b.clear();
        ref_triangle_points_a.push_back(keys[ref_obj[0].queryIdx].pt);
        ref_triangle_points_b.push_back(keys[ref_obj[0].trainIdx].pt);
        //ref_triangle.push_back(ref_obj[0]);
        ref_obj.erase(ref_obj.begin());

        int points = 1;
        vector<DMatch>::iterator it1 = ref_obj.begin();
        while (it1 != ref_obj.end() && points != 3) {
            bool found = true;
            vector<Point2f>::iterator it2 = ref_triangle_points_a.begin();
            while (it2 != ref_triangle_points_a.end()) {
                if (it2->x == keys[it1->queryIdx].pt.x && it2->y == keys[it1->queryIdx].pt.y) {
                    found = false;
                    break;
                }
                it2++;
            }
            if (found && is_near(keys[ref_obj[0].trainIdx].pt, keys[it1->trainIdx].pt, limitx, limity)) {
                ref_triangle_points_a.push_back(keys[it1->queryIdx].pt);
                ref_triangle_points_b.push_back(keys[it1->trainIdx].pt);
                obj.push_back(*it1);
                //ref_triangle.push_back(*it1);
                points++;
                it1 = ref_obj.erase(it1);
            } else {
                it1++;
            }
        }

        //obj.push_back(ref_triangle[0]);
        //obj.push_back(ref_triangle[1]);
        //obj.push_back(ref_triangle[2]);

        cout << "3 Punkte gefunden!" << endl;


        for (vector<DMatch>::iterator match = ref_obj.begin(); match != ref_obj.end();) {
            Point2f org = keys[match->queryIdx].pt;
            Point2f check = keys[match->trainIdx].pt;

            if (checkAreaQuotient(ref_triangle_points_a, ref_triangle_points_b, org, check)) {
                obj.push_back(*match);
                match = ref_obj.erase(match);
            } else {
                match++;
            }
        }

        objects.push_back(obj);
    }
    return objects;
}

void algorithmus(String PATH, int Hessian, int anzahl_Matches, int max_Distance, int limit_x, int limit_y, int darstellung)
{
    //Load Picture/Frame and save decriptors, die mittels SIFT-Detection bestimmt werden
    //Lade das Bild 2 mal, um später matches finden zu können
    //path = Pfad zum Bild
    //minHessian = Anzahl der Keypoints die berechnet werden

    String path = PATH;
    int minHessian = Hessian;

    Mat img_1 = imread(path);
    Mat img_2 = imread(path);
    Ptr<SIFT> detector = SIFT::create(minHessian);

    Mat mask = Mat::ones(img_1.size(),CV_8U);
    vector<KeyPoint> keys1, keys2;
    Mat desc1, desc2;

    detector->detectAndCompute(img_1,mask,keys1,desc1);
    detector->detectAndCompute(img_2,mask,keys2,desc2);



    //________________________________________________________________________________________________
    //FlannMatcher findet mit dem knnMatch-Algorithmus die passenden Matches
    //anzahl_matches = Anzahl der vom knnMatch am besten passenden Matches
    //max_distance = Größte Distanz die zulässig ist, um als guter Match ientifiziert zu werden

    //TODO
    //anzahl_matches variabler gestallten, damit eine beliebige Anzahl an Objekten detektiert werden kann
    //aber nicht zu viele unnötige Matches gefunden werden
    //vielleicht den algorithmus mehr mals durchlaufen lassen, mit verschiedenen anzahl an matches
    //und das beste ergebnis(bestes differenzbild) wird als bestes ergebnis interpretiert

    //TODO
    //max_distance varibel machen, damit die sensitivität des Algorithmus besser wird
    //50 ist in dem Beispiel mit den Karten okay, aber in anderen Beispielen nicht
    //auch hier mehrmals durchlaufen lassen? oder lieber einen festen wert?


    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    int anzahlMatches=anzahl_Matches;
    int maxDistance = max_Distance;
    matcher.knnMatch(desc1,desc2,matches,anzahlMatches);

    //Filter Keypoints with good matches
    vector<DMatch> good_matches;
    for(int i = 0; i<desc1.rows;i++)
    {
        for(int j = 0; j<anzahlMatches;j++)
        {
            //cout << "Distanz für Match " << i << " " << j << ": " << matches[i][j].distance << endl;
            if(matches[i][j].distance>0&&matches[i][j].distance<maxDistance)
            {
                cout << "Pushed " << j << endl;
                good_matches.push_back(matches[i][j]);
            }
        }
    }


    //______________________________________________________________________________________
    //Teile die Keypoints in Objekte auf und finde für jedes Objekt eine Homographie
    //Zunächst ein Objekt als Basisobjekt festlegen
    //Keypoints der anderen Objekte trennen


    //TODO
    //Flächenverhätnis = const bei affine Transformation
    //Problem!! Ersten 3 Punkte müssen nicht zwingend in einem Objekt liegen!!


    //Nur eines der sich wiederholenden Objekte als Basis
    vector<DMatch> ref_obj;
    ref_obj.push_back(good_matches[0]);
    Point2f ref_a = keys1[good_matches[0].queryIdx].pt;
    int limitx = limit_x, limity = limit_y;
    for(size_t i = 1; i<good_matches.size();i++){
        Point2f ref_b = keys1[good_matches[i].queryIdx].pt;

        if(is_near(ref_a,ref_b,limitx,limity)){
            ref_obj.push_back(good_matches[i]);
        }
    }

    //Splitte die Transformierte Objekte

    vector<vector<DMatch>> objects = split_objects(ref_obj,keys2,limitx,limity,--anzahlMatches);

    vector<Mat> homographies;

    for(vector<DMatch> v : objects){
        homographies.push_back(homography(ref_obj,v,keys1));
    }


    //______________________________________________________________________________________________________
    //Zeige das Ergebnis an und beende das Programm durch das Drücken auf eine beliebige Taste

    vector<Mat> img_matches;
    if(darstellung==-1)
    {
        for (int i = 0; i < objects.size(); ++i) {
            Mat im;
            drawMatches(img_1,keys1,img_2,keys2,objects[i],im);
            img_matches.push_back(im);
        }
    }
    else{
        Mat im;
        drawMatches(img_1,keys1,img_2,keys2,objects[darstellung],im);
        img_matches.push_back(im);
    }

    for (int k = 0; k < img_matches.size(); ++k) {
        imshow("Test",img_matches.at(k));
        waitKey(0);
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

bool contains(vector<Point2f> vp, Point2f p2f)
{
    bool contain = false;
    vector<Point2f>::iterator it = vp.begin();
    while(it != vp.end())
    {
        if(it->x == p2f.x && it->y == p2f.y){
            contain = true;
            break;
        }
        it++;
    }

    return contain;
}

vector<vector<DMatch>> Homography(vector<DMatch> good_matches,vector<KeyPoint> keys1, Mat* IMG1, Mat* IMG_TRANS)
{
    vector<Point2f> queryPoints, trainPoints;
    vector<DMatch>::iterator it = good_matches.begin();
    while(it!=good_matches.end())
    {
//        if(!contains(queryPoints,keys1[it->queryIdx].pt)){
            queryPoints.push_back(keys1[it->queryIdx].pt);
            trainPoints.push_back(keys1[it->trainIdx].pt);
//        }
        it++;
    }
    Mat maskinliers;
    Mat H = findHomography(queryPoints,trainPoints,CV_RANSAC,10,maskinliers,3000,0.995);
    cout << H << endl;
    cout << maskinliers.type() << endl;
    cout << maskinliers.channels() << endl;


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


    if(points.size()!=0)
    {
        perspectiveTransform(points,trans_points,H);
        cout << points.size() << " " << trans_points.size() << endl;

        for(int i = 0; i<points.size();i++){
            Point2f p_org = points.at(i); //cout << p_org.x << " " << p_org.y << endl;
            Point2f p_trans = trans_points.at(i); //cout << p_trans.x << " " << p_trans.y << endl << endl;
//            if(!(p_org.x < 0 || p_org.y < 0 || p_trans.x < 0 || p_trans.y <0))
//                IMG_TRANS->at<Vec3b>(p_trans.y,p_trans.x) = IMG1->at<Vec3b>(p_org.y,p_org.x);
        }
    }


    vector<vector<DMatch>> returnvalue;
    returnvalue.push_back(match);
    returnvalue.push_back(rest);


    return returnvalue;
}

//Mat Homography(vector<DMatch> good_matches, Mat maskinliers, vector<KeyPoint> keys1, Mat* IMG1, Mat* IMG_TRANS)
//// returns the inliers mask of the homography
//{
//    vector<Point2f> queryPoints, trainPoints;
//    vector<DMatch>::iterator it = good_matches.begin();
//    while(it!=good_matches.end())
//    {
//        if(!contains(queryPoints,keys1[it->queryIdx].pt)){
//            queryPoints.push_back(keys1[it->queryIdx].pt);
//            trainPoints.push_back(keys1[it->trainIdx].pt);
//        }
//        it++;
//    }
//    Mat maskinliers;
//    Mat H = findHomography(queryPoints,trainPoints,CV_RANSAC,10,maskinliers,3000,0.995);
//    cout << H << endl;

//    vector<DMatch> match, rest;
//    for(int i = 0; i<maskinliers.rows; i++)
//    {
//        DMatch d = good_matches.at(i);
//        if((unsigned int)maskinliers.at<uchar>(i)){
//            match.push_back(d);
//        }else{
//            rest.push_back(d);
//        }
//    }

//    vector<Point2f> points, trans_points;

//    for(DMatch m : good_matches)
//    {
//        points.push_back(keys1[m.queryIdx].pt);
//    }


//    if(points.size()!=0)
//    {
//        perspectiveTransform(points,trans_points,H);
//        cout << points.size() << " " << trans_points.size() << endl;

//        for(int i = 0; i<points.size();i++){
//            Point2f p_org = points.at(i); //cout << p_org.x << " " << p_org.y << endl;
//            Point2f p_trans = trans_points.at(i); //cout << p_trans.x << " " << p_trans.y << endl << endl;
//            if(!(p_org.x < 0 || p_org.y < 0 || p_trans.x < 0 || p_trans.y <0))
//                IMG_TRANS->at<Vec3b>(p_trans.y,p_trans.x) = IMG1->at<Vec3b>(p_org.y,p_org.x);
//        }
//    }


//    vector<vector<DMatch>> returnvalue;
//    returnvalue.push_back(match);
//    returnvalue.push_back(rest);


//    return returnvalue;
//}


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
            cout <<  matches[i][j].distance  << endl;
            if(matches[i][j].distance<min && matches[i][j].distance!=0)
                min = matches[i][j].distance;
            if(matches[i][j].distance>max && matches[i][j].distance!=0)
                max = matches[i][j].distance;
        }
    }
    cout << "Min: " << min << endl;
    cout << "Max: " << max << endl;

    for(int i = 0; i < desc1.rows; i++){
        for(int j = 0; j < anzahlMatches; j++){
//            if(matches[i][j].distance <= 3*min)
                good_matches.push_back(matches[i][j]);
        }
    }

    for(DMatch m : good_matches){
        if(!isEqual(m,keys1))
            filtered_matches.push_back(m);
    }

//    Mat img_good_matches;
//    drawMatches(img_1,keys1,img_2,keys2,good_matches,img_good_matches );
//    namedWindow("GoodMatches", CV_WINDOW_KEEPRATIO);
//    resizeWindow("GoodMatches", 800, 800);
//    imshow("GoodMatches",img_good_matches);
//    waitKey(0);

    Mat img_filtered_matches;
    drawMatches(img_1,keys1,img_2,keys2,filtered_matches,img_filtered_matches );
    namedWindow("ReducedMatches", CV_WINDOW_KEEPRATIO);
    resizeWindow("ReducedMatches", 800, 800);
    imshow("ReducedMatches",img_filtered_matches);
    waitKey(0);




    Mat img_trans = Mat::zeros(img_1.size(),img_1.type());
    vector<vector<DMatch>> split = Homography(filtered_matches,keys1,&img_1, &img_2);

    //drawMatches(img_1,keys1,img_2,keys2,split[0],img_matches);
    //imshow("Matches",img_matches);
    //waitKey(0);

    //drawMatches(img_1,keys1,img_2,keys2,split[1],img_matches2);
    //imshow("Rest",img_matches2);
    //waitKey(0);

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
    }while(split[1].begin()!=split[1].end()&&split[1].size()>3);

    Mat diff_img;
    imshow("Bild2",img_2);
    imshow("Bild1",img_1);
    absdiff(img_1,img_2,diff_img);


    imshow("Bild2",img_2);
    imshow("Transimg",diff_img);
    waitKey(0);
}
