#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <random>

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