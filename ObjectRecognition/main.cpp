//
//  main.cpp
//  ObjectRecognition
//
//  Created by Vamsi Mocherla on 3/16/14.
//  Copyright (c) 2014 VamsiMocherla. All rights reserved.
//

#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    Mat object = imread("mascot.JPG");
    if(!object.data)
    {
        cout << "[ERROR] Cannot read image" << endl;
        return -1;
    }
    
    // detect key points in the image using SURF
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> kpObject;
    
    detector.detect(object, kpObject);
    
    // compute feature descriptors
    SurfDescriptorExtractor extractor;
    Mat desObject;
    
    extractor.compute( object, kpObject, desObject );
    
    // get the corners of the object
    vector<Point2f> objCorners(4);
    
    objCorners[0] = cvPoint(0, 0);
    objCorners[1] = cvPoint(object.cols, 0);
    objCorners[2] = cvPoint(object.cols, object.rows);
    objCorners[3] = cvPoint(0, object.rows);
    
    // capture video from webcam
    VideoCapture input(0);
    
    if(!input.isOpened())
	{
		cout << "[ERROR] Cannot open webcam" << endl;
		return -1;
	}
    
	cout << "[MESSAGE]: CAPTURING VIDEO FROM WEBCAM" << endl;
    
    
    Mat inputFrame;
    
    int key = 0;
    while(key != 27)
    {
		// capture each frame of the video
		input >> inputFrame;
		if(inputFrame.empty())
			break;
        // resize the image for faster processing
        resize(inputFrame, inputFrame, Size(), 0.5, 0.5, INTER_LINEAR);
        
        // detect key points in the input frame
        vector<KeyPoint> kpFrame;
        detector.detect(inputFrame, kpFrame);
        
        // extract feature descriptors for the detected key points
        Mat desFrame;
        extractor.compute(inputFrame, kpFrame, desFrame);
        if(desFrame.empty())
            continue;
        
        // match the key points with object
        FlannBasedMatcher matcher;
        vector< vector <DMatch> > matches;
        matcher.knnMatch(desObject, desFrame, matches, 2);
        
        // compute the good matches among the matched key points
        vector<DMatch> goodMatches;
        for(int i=0; i<desObject.rows; i++)
        {
            if(matches[i][0].distance < 0.6 * matches[i][1].distance)
            {
                goodMatches.push_back(matches[i][0]);
            }
        }
        
        // draw the good matches
        Mat imageMatches;
        drawMatches(object, kpObject, inputFrame, kpFrame, goodMatches, imageMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(),  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        if(goodMatches.size() >= 4)
        {
            vector<Point2f> obj;
            vector<Point2f> scene;
            
            for( int i = 0; i < goodMatches.size(); i++ )
            {
                // get the keypoints from the good matches
                obj.push_back( kpObject[ goodMatches[i].queryIdx ].pt );
                scene.push_back( kpFrame[ goodMatches[i].trainIdx ].pt );
            }
            
            Mat H;
            H = findHomography(obj, scene);
            
            vector<Point2f> sceneCorners(4);
            perspectiveTransform( objCorners, sceneCorners, H);
            
            // draw lines between the corners (the mapped object in the scene image )
            line(imageMatches, sceneCorners[0]+Point2f(object.cols, 0), sceneCorners[1]+Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(imageMatches, sceneCorners[1]+Point2f(object.cols, 0), sceneCorners[2]+Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(imageMatches, sceneCorners[2]+Point2f(object.cols, 0), sceneCorners[3]+Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(imageMatches, sceneCorners[3]+Point2f(object.cols, 0), sceneCorners[0]+Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
        }
        imshow("Matches", imageMatches);
        key = waitKey(1);
    }
    
    return 0;
}

