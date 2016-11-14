////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

constexpr double const_pi() { return std::atan(1)*4; }

typedef std::chrono::high_resolution_clock::time_point time_var;
template<typename F, typename... Args>
uint64_t fun_time(F func, Args&&... args){
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    func(std::forward<Args>(args)...);
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

void d3_transform(const Mat &src_image, Mat &output, const std::array< std::array<int, 2>, 4> & trans_cord, int precision=100) {

max win: -76--150
514:280-470:156-286:204-320:333


                (1)470:156   
(2)286:204
                    (0)514:280
    (3)320:333

	/// Set source points in the corners of image
	std::array<Point2f, 4> src_cordinate = {{
			Point2f( 0,0 ),			        Point2f( src_image.cols, 0 ),
			Point2f( 0, src_image.rows ),	Point2f( src_image.cols , src_image.rows )
		}
	};
	// transform cordinates to pionts
	std::array<Point2f, 4> dst_points = {{
			Point2f( trans_cord[0][0], trans_cord[0][1]  ),
			Point2f( trans_cord[3][0], trans_cord[3][1] ),
			Point2f( trans_cord[1][0], trans_cord[1][1] ),
			Point2f( trans_cord[2][0], trans_cord[2][1] ),
		}
	};

	// absolute to relative cordinate
	for(auto &i : dst_points) {
		i.x -= std::min(trans_cord[1][0],trans_cord[0][0]);
		i.y -= std::min(trans_cord[0][1],trans_cord[3][1]);
		std::cout << "x" << i.x << "y" << i.y << '\n';
	}

	Mat warp_mat( 2, 3, CV_32FC1 );
	/// Set the dst image the same type and size as src
	Mat warp_dst;

	int max_height = std::max(trans_cord[1][1]-trans_cord[3][1], trans_cord[2][1]-trans_cord[0][1]);
	int max_weight = std::max(trans_cord[3][0]-trans_cord[1][0], trans_cord[2][0]-trans_cord[0][0]);
	std::cout << "\nmax win: " <<  max_height << "-" <<   max_weight << '\n';
	std::cout   << trans_cord[0][0] << ":" << trans_cord[0][1] << "-" << trans_cord[1][0] << ":" << trans_cord[1][1] << "-"
	            << trans_cord[2][0] << ":" << trans_cord[2][1] << "-" << trans_cord[3][0] << ":" << trans_cord[3][1] << '\n';

	if(max_height < 0 || max_weight < 0) {
		std::cout << "bad MAX" << std::endl;
		return;
	}

	warp_dst = Mat::zeros( max_height, max_weight, src_image.type() );
	warp_mat = getPerspectiveTransform(src_cordinate.data(), dst_points.data());
	warpPerspective(src_image,warp_dst, warp_mat, warp_dst.size() );

	output = warp_dst;
}


struct O_moments_data {
	double dM01;
	double dM10;
	double dArea;
	double dM01_by_dArea() const {
		return dM01/dArea;
	}
	double dM10_by_dArea() const {
		return dM10/dArea;
	}
};

O_moments_data calculate_oMoments_img (const Mat &imgThresholded) {
	Moments oMoments = moments(imgThresholded);
	return	{	oMoments.m01,
	            oMoments.m10,
	            oMoments.m00
	       };
}

O_moments_data calculate_oMoments_circles (const std::vector<Vec3f> &circles) {
    
    if (circles.size() < 4) {
        throw std::invalid_argument("calculate_oMoments_circles needs at least 4 circles");
    }
    // calculate for only 4 circles - performance.
    std::vector<cv::Point> points(4);
    points[0].x = circles[0][0] ; points[0].y = circles[0][1];
    points[1].x = circles[1][0] ; points[1].y = circles[1][1];
    points[2].x = circles[2][0] ; points[2].y = circles[2][1];
    points[3].x = circles[3][0] ; points[3].y = circles[3][1];

    Moments oMoments = moments(points, false);
	return	{	oMoments.m01,
	            oMoments.m10,
	            oMoments.m00
	       };
}

void draw_line_on (Mat & imgLines, O_moments_data & omoments) {
	static int iLastX = -1;
	static int iLastY = -1;

	bool pos_ok = false;
	// if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
	if (omoments.dArea > 10000) {
		//calculate the position of the ball
		int posX = omoments.dM10 / omoments.dArea;
		int posY = omoments.dM01 / omoments.dArea;

		if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
			//Draw a red line from the previous point to the current point
			line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
			pos_ok = true;
		}

		iLastX = posX;
		iLastY = posY;

		//std::cout << "posX:" << posX << "\t posY:" <<  posY << "\t iLastX:" << iLastX << "\t iLastY:" << iLastY;
	}
}

struct Dct_circle_param {
	int precision;
	int dp;
	int minDist;
	int param1;
	int param2;
	int minRadius;
	int maxRadius;
};

// copy because we don't want to modify original Mat object
std::vector<Vec3f> detect_circles (Mat image, Dct_circle_param cparam) {

	int method = CV_HOUGH_GRADIENT;

	GaussianBlur( image, image, Size(9, 9), 2, 2 );
	//GaussianBlur( image, image, Size(15, 15), 6, 6 );
	vector<Vec3f> circles;

	/// +1 to prevent non positive number
	HoughCircles(image, circles, method, cparam.dp+1, cparam.minDist/cparam.precision+1, cparam.param1/cparam.precision+1, cparam.param2/cparam.precision+1, cparam.minRadius, cparam.maxRadius );

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
        
void fit_to_4cirles(const Mat &src_image, Mat &output, const std::vector<Vec3f> &circles, const O_moments_data omoments) {
    // this condition could be checked before for avoid copy
    if(circles.size() != 4){
        throw std::invalid_argument("get_fitted_to_4cirles exactly needs 4 circles");
        
    }

    // determine the correct quadrant of 4 circles
    std::map<float, Point2f> point_map =
    [&circles, &omoments] {
        std::map<float, Point2f> loc_map;
        for(const auto &vec3f : circles) {
            int relative_X = cvRound(vec3f[0] -  omoments.dM10_by_dArea());
            int relative_Y = cvRound(vec3f[1] -  omoments.dM01_by_dArea());
            std::cout << "relativeY:" << relative_Y << std::endl;
            double tan_result = std::atan2(relative_Y, relative_X);
            // rotate by pi/2 
            std::cout << "Rotate " << tan_result << " to ";
            (tan_result+const_pi()/2) > const_pi() ? tan_result = -const_pi() + (tan_result - const_pi()/2)  : tan_result += const_pi()/2;
            std::cout  << tan_result << '\n';
            // fill map
            loc_map.emplace(tan_result, Point2f(relative_X, relative_Y));
        }
        return loc_map;
    } ();

    std::cout << "points [";
    for (const auto &ele : point_map) {
    	std::cout << ele.first << " : " << ele.second << " ";
    }
    std::cout << '\n';

    // reverse iterator because the first map key is in bottom left quadrant ~(-2:-3)
    bool rec_ok = true;
    {
        // poor test
        auto it_test = point_map.begin();
        if(it_test->first >= 0) rec_ok = false;
        it_test++;
        if(it_test->first > 0) rec_ok = false;
        it_test++;
        if(it_test->first < 0) rec_ok = false;
        it_test++;
        if(it_test->first <= 0) rec_ok = false;
    }

    std::array< std::array<int,2>, 4> trans_cord;
    int point_nu = 0;
    for (auto it = point_map.rbegin() ; it != point_map.rend() ; ++it) {
        trans_cord.at(point_nu)[0] = it->second.x + omoments.dM10_by_dArea() ; // back to absolute
        trans_cord.at(point_nu)[1] = it->second.y + omoments.dM01_by_dArea() ;
        point_nu++;
    }

    {
        // transform cordinate test
        //if( std::max(trans_cord[1][1]-trans_cord[3][1], trans_cord[2][1]-trans_cord[0][1]) <= 5 ) rec_ok = false;
        //if( std::max(trans_cord[3][0]-trans_cord[1][0], trans_cord[2][0]-trans_cord[0][0]) <= 5 ) rec_ok = false;
    }
    
    if(rec_ok) {
        d3_transform(src_image, output, trans_cord);
    } else {
        std::cout << "REC FALSE" << '\n';
    } 
}

/// started from http://jepsonsblog.blogspot.com/2012/10/overlay-transparent-image-in-opencv.html
void overlay_image(const Mat &background, const Mat &foreground, Mat &output,const O_moments_data &omoments) {
        
    Point2i location = Point2i (static_cast<int>(omoments.dM10_by_dArea() - foreground.cols/2) ,
                                static_cast<int>(omoments.dM01_by_dArea() - foreground.rows/2));

    // check if whether the coordinates are possible to draw
	int fore_weight = foreground.cols;
	int fore_height	= foreground.rows;
	if (location.x < 0 ||
	    location.y < 0 ||
	    location.x - fore_weight/2-1 <= 0 ||
	    location.y - fore_height/2-1 <= 0 ||
	    location.x + fore_weight/2+1 >= background.cols ||
	    location.y + fore_height/2+1 >= background.rows) {


        std::cout << "overlayImage fail " << location.x << ":" << location.y << std::endl;
		//return;
        throw std::invalid_argument("get_fitted_to_4cirles exactly needs 4 circles");
	}

	background.copyTo(output);

	// start at the row indicated by location, or at row 0 if location.y is negative.
	for(int y = std::max(location.y , 0); y < background.rows; ++y) {
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if(fY >= foreground.rows)
			break;

		// start at the column indicated by location,

		// or at column 0 if location.x is negative.
		for(int x = std::max(location.x, 0); x < background.cols; ++x) {
			int fX = x - location.x; // because of the translation.

			// we are done with this row if the column is outside of the foreground image.
			if(fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity = static_cast<double>(foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;
			//double opacity = 1;

			// and now combine the background and foreground pixel, using the opacity,

			// but only if opacity > 0.
			for(int c = 0; opacity > 0 && c < output.channels(); ++c) {
				unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] = backgroundPx * (1.-opacity) + foregroundPx * opacity;
			}
		}
	}
}
int main( int argc, char** argv ) {

	Mat img_camera;
	VideoCapture cap(0); //capture the video from web cam
	if ( !cap.isOpened() ) { // if not success, exit program
		std::cout << "Cannot open the web cam" << std::endl;
		return -1;
	}

	Mat logo;
	logo = imread("./media/example2.png", CV_LOAD_IMAGE_UNCHANGED);   // Read the file

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

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    // HSV color threshold parameters
	int iLowH = 37;
	int iHighH = 179;

	int iLowS = 40;
	int iHighS = 255;

	int iLowV = 125;
	int iHighV = 255;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    // circles detection parameters
	int precission = 10;
    int dp = 0;
    int minDist = 100;
    int param1 = 100; int param2 = 150;
    int minRadius = 5; int maxRadius =30;

    Dct_circle_param cparam = {precission, dp, minDist, param1, param2, minRadius, maxRadius};

	cvCreateTrackbar("dp", "Control", &cparam.dp, 15);
	cvCreateTrackbar("minDist", "Control", &cparam.minDist, 500);
	cvCreateTrackbar("param1", "Control", &cparam.param1, 500);
	cvCreateTrackbar("param2", "Control", &cparam.param2, 500);

	cvCreateTrackbar("minradius", "Control", &cparam.minRadius, 70);
	cvCreateTrackbar("maxradius", "Control", &cparam.maxRadius, 70);


	//Capture a temporary image from the camera
	Mat imgTmp;
	cap.read(imgTmp);

	//Create a black image with the size as the camera output
	Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;
	Mat logo_transformed(logo);

    uint64_t fit_circles,overlay_img;
	while (true) {
try {
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
		//GaussianBlur( imgThresholded, imgThresholded, Size(9, 9), 2, 2 );

		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );

		//Calculate the moments of the thresholded image
		O_moments_data omoments = calculate_oMoments_img(imgThresholded);
		
        // Drawing line
        //draw_line_on (imgLines, omoments);
		//img_camera = img_camera + imgLines;


		imshow("Thresholded Image", imgThresholded); //show the thresholded image

		std::vector<Vec3f> circles = detect_circles(imgThresholded, cparam);
		std::cout << "circles: " << circles.size() << " ";
       

        if(circles.size() == 4) {
            omoments = calculate_oMoments_circles(circles);
            fit_circles = fun_time(fit_to_4cirles, logo, logo_transformed, circles, omoments);
        } 
        //else {
        //    waitKey(10);
        //    continue;
        //}
		
		// get frame from sample avi
		//cap_sample >> sample_frame;

		//if(sample_frame.empty()) {
		//	cout << "empty -  Cannot read a frame from sample avi" << endl;
		//	cout << "empty -  restarting video" << endl;
		//	cap_sample.set(CV_CAP_PROP_POS_FRAMES,0);
		//	continue;
		//}
		
		draw_circle_on(img_camera, circles);

        Mat overlay_result;
        
        overlay_img = fun_time(overlay_image, img_camera, logo_transformed, overlay_result, omoments);
        imshow("Project", overlay_result );
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		if (waitKey(30) == 27) { //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			cout << "esc key is pressed by user" << endl;
			break;
		}
		float fps = 1000./std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		std::cout   << " [ fps:" << setprecision(4) <<  fps << " ] fic_crc:" << fit_circles << "us, overlay_img:"<< overlay_img
                    << "us, rest:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << '\n';
}
catch(const std::exception &) { }
	}
	return 0;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



