#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;

int main() {
    // Create a sample dataset for training
    Mat samples = (Mat_<float>(5, 2) << 2.0, 3.0,
                                       4.0, 6.0,
                                       5.0, 4.0,
                                       7.0, 8.0,
                                       8.0, 6.0);

    Mat labels = (Mat_<int>(5, 1) << 0, 1, 1, 1, 0); // Labels for the dataset

    // Train the SVM model
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->train(samples, ROW_SAMPLE, labels);

    // Test the model with a new sample
    Mat testSample = (Mat_<float>(1, 2) << 6.0, 5.0);
    float prediction = svm->predict(testSample);

    if (prediction == 0)
        std::cout << "Predicted class: 0 (Class A)" << std::endl;
    else if (prediction == 1)
        std::cout << "Predicted class: 1 (Class B)" << std::endl;
    else
        std::cout << "Unknown class" << std::endl;

    return 0;
}
