//STD
#include <fstream>
#include <sstream>
#include <iostream>

//OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//ROS2
#include <rclcpp/rclcpp.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class YoloObjectDetection : public rclcpp::Node
{
public:
    YoloObjectDetection()
    : Node("yolo_object_detection")
    {
        parse_parameters_();
        RCLCPP_INFO(this->get_logger(), "Parameters parsed");
        run_();
        RCLCPP_INFO(this->get_logger(), "Run successful");
    }
private:

    //Initialize ROS2 parameters
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    string device_;
    string input_type_;
    string input_path_;
    bool write_output_;
    string output_file_;
    string classesFile;
    String modelConfiguration;
    String modelWeights;

    void parse_parameters_();
    void run_();
    void postprocess(const Mat& frame, const vector<Mat>& outs);
    void drawPred(int idx, int classId, float conf, int left, int top, int right, int bottom, const Mat& frame);
    vector<String> getOutputsNames(const Net& net);
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloObjectDetection>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}