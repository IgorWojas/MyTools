#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <random>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

typedef Eigen::MatrixXd Matrix;

int getDay() {
    int out;
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTimeInfo;
    if (localtime_s(&localTimeInfo, &currentTime) == 0) {
        out = localTimeInfo.tm_mday;
    }
    else {
        std::cerr << "Error converting time_t to tm." << std::endl;
    }
    return out;
}

int getMonth() {
    int out;
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTimeInfo;
    if (localtime_s(&localTimeInfo, &currentTime) == 0) {
        out = localTimeInfo.tm_mon + 1;
    }
    else {
        std::cerr << "Error converting time_t to tm." << std::endl;
    }
    return out;
}

int getYear() {
    int out;
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTimeInfo;
    if (localtime_s(&localTimeInfo, &currentTime) == 0) {
        out = localTimeInfo.tm_year + 1900;
    }
    else {
        std::cerr << "Error converting time_t to tm." << std::endl;
    }
    return out;
}

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
void loopWindow(std::string winName, int width, int height, Mat in) {
    namedWindow(winName, 0);
    resizeWindow(winName, width, height);
    imshow(winName, in);
    waitKey(1);
}

// live window preview, window position (x, y)
void loopWindow(std::string winName, int width, int height, Mat in, int posx, int posy) {
    namedWindow(winName, 0);
    resizeWindow(winName, width, height);
    moveWindow(winName, posx, posy);
    imshow(winName, in);
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

// Timer t1 > t1.stop()
class Timer {
public:
    Timer() {
        startTimePoint = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto endTimePoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();
        auto duration = end - start;
        std::cout << "Elapsed time: " << duration * 0.001 << "ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};

// fast : flat
std::vector<double> mat2vec(cv::Mat vin) {
    cv::Mat flat = vin.reshape(1, vin.total() * vin.channels());
    std::vector<double> out = vin.isContinuous() ? flat : flat.clone();
    return out;
}

// CV_64F , fast : buffer
Mat vector2mat(const std::vector<double>& data, int rows, int cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Size of data vector does not match rows and cols parameters.");
    }
    cv::Mat mat(rows, cols, CV_64F);
    memcpy(mat.data, data.data(), data.size() * sizeof(double));
    return mat;
}

// Create a vector of X Eigen Matrices of size X - Y with all-zeros but first column is 1 Then each cosecutive matrix has 1's in the next column 
std::vector<Matrix> makeTrainMatrixC(int x, int y) {
    std::vector<Matrix> matrices;
    for (int i = 0; i < x; ++i) {
        Matrix matrix = Matrix::Zero(y, x);
        matrix.col(i).setOnes();
        matrices.push_back(matrix);
    }
    return matrices;
}

// Create a vector of Y Eigen Matrices of size X - Y with all-zeros but first row is 1 Then each cosecutive matrix has 1's in the next row
std::vector<Matrix> makeTrainMatrixR(int x, int y) {
    std::vector<Matrix> matrices;
    for (int i = 0; i < y; ++i) {
        Matrix matrix = Matrix::Zero(y, x);
        matrix.row(i).setOnes();
        matrices.push_back(matrix);
    }
    return matrices;
}
// returns Matrix(width, height) with random double values(min, max)
Matrix makeSomeNoise(int width, int height, int min, int max) {
    Matrix noise(width, height);
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            noise(w, h) = getRandomDouble(min, max);
        }
    }
    return noise;
}

// Returns the column index of max value in a matrix
int getMaxCol(Matrix in) {
    Eigen::Index maxRow, maxCol;
    double maxValue = in.maxCoeff(&maxRow, &maxCol);
    return maxCol;
}

// Returns the row index of max value in a matrix
int getMaxRow(Matrix in) {
    Eigen::Index maxRow, maxCol;
    double maxValue = in.maxCoeff(&maxRow, &maxCol);
    return maxRow;
}




/////////////////////////////DEBUG//////////////////////////////

//Screamer !!!!!!!!!!!!AAAAAAAA
void a(int a) { std::cout << "**********(( " << a << " ))**********" << std::endl; }

/////////////////////////////////////////////////////////////////





////////////////////PROTOTYPES/////////////////////////////////////

class Monitor { // coœ tu z paddigniem jest nie tak ale jakotako dziala
    // WeŸ to wszystko na kartce policz a nie sie bawisz...
    // i jeszcze musi byc jakos podawana rozdzielczosc obrazu do loopwindow
private:
    int outW = 0;
    int outH = 0;
    int boxW = 0; //stop
    int startx = 0; //900
    int starty = 0; // -230
    int winWidth;
    int winHeight;
    int maxHori;
    int maxVert;
    int xpadding = 7;
    int ypadding = 32;
    int screenWidth = 1920;
    int screenHeight = 1080;
    vector<Matrix> wins;
public:
    // max windows horizontal, max windows vertical, window width, window height
    Monitor(int numHori, int numVert, int width, int height, int outputW, int outputH) {
        outW = outputW;
        outH = outputH;
        maxHori = numHori;
        maxVert = numVert;
        winWidth = width;
        winHeight = height;
        boxW = (maxHori * winWidth) + ((maxHori + 1) * xpadding);
        startx = screenWidth - boxW;
        //starty = screenHeight - (maxVert * (winHeight + 30)) - (maxVert * padding);
    }
    
    

    void addWin(Matrix in) {
        wins.push_back(in);
    }

    void display() {
        int counter = 0;
        for (int hori = maxHori; hori > 0; hori--) {
            for (int vert = 0; vert < maxVert; vert++) {
                if (counter < (wins.size())) {
                    string winName = to_string(counter);
                    loopWindow(winName, winWidth, winHeight, eigen2mat(wins[counter], outW, outH), startx + (hori * xpadding)+((hori - 1) * winWidth), (vert * (winHeight + ypadding)) + starty);
                    //cout << "counter = " << counter << endl;
                    //waitKey(0);
                    counter++;
                }
            }
        }
        clear();
    }
    void clear() {
        wins.clear();
    }
};



//////////////////// bckp Monitor, to juz dziala tylko zle pozycje sa //////////////

//class Monitor { // coœ tu z paddigniem jest nie tak ale jakotako dziala
//public:
//
//    // max windows horizontal, max windows vertical, window width, window height
//    Monitor(int numHori, int numVert, int width, int height) {
//        maxHori = numHori;
//        maxVert = numVert;
//        winWidth = width;
//        winHeight = height;
//        startx = screenWidth - ((maxHori + 1) * winWidth) - xpadding; //(maxHori * xpadding)
//        //starty = screenHeight - (maxVert * (winHeight + 30)) - (maxVert * padding);
//    }
//
//
//    int startx = 0; //900
//    int starty = 0; // -230
//    int winWidth;
//    int winHeight;
//    int maxHori;
//    int maxVert;
//    int xpadding = 5;
//    int ypadding = 32;
//    int screenWidth = 1920;
//    int screenHeight = 1080;
//    vector<Matrix> wins;
//    void addWin(Matrix in) {
//        wins.push_back(in);
//    }
//    void display() {
//        int counter = 0;
//        for (int hori = maxHori; hori > 0; hori--) {
//            for (int vert = 0; vert < maxVert; vert++) {
//                if (counter < (wins.size())) {
//                    string winName = to_string(counter);
//                    loopWindow(winName, winWidth, winHeight, eigen2mat(wins[counter], 20, 20), (hori * (winWidth)) + startx - xpadding, (vert * (winHeight + ypadding)) + starty);
//                    //cout << "counter = " << counter << endl;
//                    //waitKey(0);
//                    counter++;
//                }
//            }
//        }
//    }
//};
