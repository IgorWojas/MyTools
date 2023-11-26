#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <random>
#include <chrono>

using namespace cv;

typedef Eigen::MatrixXd Matrix;

// CV_64F
Mat eigen2mat(Matrix in, int xi, int yi) {
    int counter = 0;
    Mat ret(xi, yi, CV_64F);
    for (int x = 0; x < xi; x++) {
        for (int y = 0; y < yi; y++) {
            ret.at<double>(y, x) = in(counter);
            counter++;
        }
    }
    return ret;
}

// Flat
Matrix mat2eigen(Mat in) {
    Matrix ret((in.cols * in.rows), 1);
    int counter = 0;
    for (int x = 0; x < in.cols; x++) {
        for (int y = 0; y < in.rows; y++) {
            ret(counter, 0) = in.at<double>(y, x);
            counter++;
        }
    }
    return ret;
}

// mt19937
double getRandomDouble(int min, int max) {
    std::random_device rd;
    std::mt19937 gen1(rd());
    std::uniform_real_distribution<> los(min, max); 
    return los(gen1);
}

// mt19937
int getRandomInt(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// live preview window
void loopWindow(int width, int height, Mat in) {
    namedWindow("tester", 0);
    resizeWindow("tester", width, height);
    imshow("tester", in);
    waitKey(1);
}

// path.bmp, GREYSCALE,  CV_64F, 1/255
Mat prepImg(std::string path) {     
    Mat mat = cv::imread(path, IMREAD_GRAYSCALE);
    mat.convertTo(mat, CV_64F);
    mat /= 255;
    return mat;
}

// Create flat eigen matrix from bmp path
Matrix path2eigen(std::string path) {
    Matrix ret = mat2eigen(prepImg(path));
    return ret;
}

// Timer zegar > zegar.startTimer() > zegar.stopTimer()
class Timer {
public:
    void startTimer() {
        startTimePoint = std::chrono::high_resolution_clock::now();
    }

    void stopTimer() {
        auto endTimePoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();
        auto duration = end - start;
        std::cout << "Elapsed time: " << duration * 0.001 << "ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};
