////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <chrono>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat put_on_image(Mat &output, Mat &img_to_put, int pos_x, int pos_y );
std::vector<Vec3f> detect_circles (Mat image);
void draw_circle_on(Mat &image, std::vector<Vec3f> &circles);

int main( int argc, char** argv ) {
        
	Mat img_camera;
    VideoCapture cap(0); //capture the video from web cam
    if ( !cap.isOpened() ) { // if not success, exit program
         std::cout << "Cannot open the web cam" << std::endl;
         return -1;
    }

	Mat logo;
//    logo = imread("example.png", CV_IMWRITE_PNG_COMPRESSION);   // Read the file
    logo = imread("./media/example.png", CV_IMWRITE_JPEG_QUALITY);   // Read the file

    if ( !logo.data ) {
       	std::cout <<  "Could not open or find the logo" << std::endl ;
       	return -1;
    }

	std::string sample_video_filename("./media/output.avi");
	Mat sample_frame;
    VideoCapture cap_sample(sample_video_filename);
    if(!cap_sample.isOpened()) {
        std::cout << "Error can't find the file: " << sample_video_filename << std::endl;
    }

	/*********************************************************************************
	// example 
	namedWindow( "w", 1);
    while(1){
		cap_sample >> sample_frame;
		if(sample_frame.empty()) {
             cout << "empty -  Cannot read a frame from sample avi" << endl;
             cout << "empty -  restarting video" << endl;
			 cap_sample.set(CV_CAP_PROP_POS_FRAMES,0);
			 continue;
		}
			imshow("w",sample_frame);
			waitKey(20); // waits to display frame
	}
	*////*****************************************************************************


    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

 	int iLowH = 0;
 	int iHighH = 179;

 	int iLowS = 0; 
 	int iHighS = 255;

 	int iLowV = 0;
 	int iHighV = 255;

 	//Create trackbars in "Control" window
 	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
 	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

 	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
 	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

 	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
 	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

 	int iLastX = -1; 
 	int iLastY = -1;

 	//Capture a temporary image from the camera
 	Mat imgTmp;
 	cap.read(imgTmp); 

 	//Create a black image with the size as the camera output
 	Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;

    while (true) {
		
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();



        bool bSuccess = cap.read(img_camera); // read a new frame from video
        if (!bSuccess) { //if not success, break loop
             cout << "Cannot read a frame from video stream" << endl;
             break;
		}


		Mat imgHSV;

		cvtColor(img_camera, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
 
		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
		  
		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

  		//Calculate the moments of the thresholded image
  		Moments oMoments = moments(imgThresholded);

  		double dM01 = oMoments.m01;
  		double dM10 = oMoments.m10;
  		double dArea = oMoments.m00;

		std::cout << dM01 << " - " << dM10 << " - " << dArea << '\t';


/*
		bool pos_ok = false;
  		// if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
  		if (dArea > 10000) {
   			//calculate the position of the ball
   			int posX = dM10 / dArea;
   			int posY = dM01 / dArea;        
        
   			if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
    		//Draw a red line from the previous point to the current point
    			line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
				pos_ok = true;
   			}

   			iLastX = posX;
   			iLastY = posY;


			std::cout << "posX:" << posX << "\t posY:" <<  posY << "\t iLastX:" << iLastX << "\t iLastY:" << iLastY;
  		}
*/
		std::vector<Vec3f> circles = detect_circles(imgThresholded);
		draw_circle_on(img_camera, circles);

		imshow("Thresholded Image", imgThresholded); //show the thresholded image
//		img_camera = img_camera + imgLines;

    	// get frame from sample avi


		cap_sample >> sample_frame;
		if(sample_frame.empty()) {
             cout << "empty -  Cannot read a frame from sample avi" << endl;
             cout << "empty -  restarting video" << endl;
			 cap_sample.set(CV_CAP_PROP_POS_FRAMES,0);
			 continue;
		}


		

		imshow("Project", put_on_image(img_camera, sample_frame, dM10/dArea, dM01/dArea) ); //show the original image
		
		if (waitKey(30) == 27) { //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            cout << "esc key is pressed by user" << endl;
            break; 
		}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		float fps = 1000./std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	
		std::cout << " [ fps:" << setprecision(3) <<  fps << " ]" << '\n'; 
		//		  << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
		//		  << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" << std::flush;
	}
	return 0;

}

// copy because we don't want to modify original Mat object
std::vector<Vec3f> detect_circles (Mat image) {
	
	int method = CV_HOUGH_GRADIENT;
	double dp = 2;
	double minDist = image.rows/4; 
	double param1=20;
	double param2=10;
	int minRadius=0;
	int maxRadius=20;

	GaussianBlur( image, image, Size(9, 9), 2, 2 );
	vector<Vec3f> circles;
	HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius );
	
	return circles;
}

void draw_circle_on(Mat &image, std::vector<Vec3f> &circles) {
    // change to for-each
	for( size_t i = 0; i < circles.size(); i++ ) {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // draw the circle center
         circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // draw the circle outline
         circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
}

Mat put_on_image(Mat &output, Mat &img_to_put, int pos_x, int pos_y ) {

	bool pos_ok = false;
	int img_weight  = img_to_put.cols;
	int img_height	= img_to_put.rows;
	
	Mat dst(output.rows, output.cols, CV_8UC3,  cv::Scalar(0,0,0));
	output.copyTo(dst);
	if (pos_x < 0 ||
		pos_y < 0 ||
		pos_x - img_weight/2-1 <= 0 || 
		pos_y - img_height/2-1 <= 0 || 
		pos_x + img_weight/2+1 >= output.cols || 
		pos_y + img_height/2+1 >= output.rows) {
	
		return dst;
	}
	
	std::cout << "X: " <<  pos_x << "Y: "  << pos_y << '\n';

	cv::Rect roi(cv::Rect(pos_x-img_weight/2, pos_y-img_height/2, img_weight, img_height));
	cv::Mat targetROI = dst(roi);
	img_to_put.copyTo(targetROI);
	targetROI = dst(cv::Rect(0, 0, img_to_put.cols, img_to_put.rows));

	return dst;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
