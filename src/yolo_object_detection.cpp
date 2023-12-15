//STD
#include <fstream>
#include <sstream>
#include <iostream>

//OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//ROS2
//#include <rmw/types.h>
#include "yolo_object_detection.h"

using namespace cv;
using namespace dnn;
using namespace std;

void YoloObjectDetection::parse_parameters_()
{   
    //Declare ROS2 parameters
    this->declare_parameter("conf_threshold");
    this->declare_parameter("nms_threshold");
    this->declare_parameter("inp_width");
    this->declare_parameter("inp_height");
    this->declare_parameter("device");
    this->declare_parameter("input_type");
    this->declare_parameter("input_path");
    this->declare_parameter("write_output");
    this->declare_parameter("output_file");
    this->declare_parameter("classes_file");
    this->declare_parameter("model_configuration");
    this->declare_parameter("model_weights");
    
    //Get parameters
    this->get_parameter("conf_threshold", confThreshold);
    this->get_parameter("nms_threshold", nmsThreshold);
    this->get_parameter("inp_width", inpWidth);
    this->get_parameter("inp_height", inpHeight);
    this->get_parameter("device", device_);
    this->get_parameter("input_type", input_type_);
    this->get_parameter("input_path", input_path_);
    this->get_parameter("write_output", write_output_);
    this->get_parameter("output_file", output_file_);
    this->get_parameter("classes_file", classesFile);
    this->get_parameter("model_configuration", modelConfiguration);
    this->get_parameter("model_weights", modelWeights);

    //Check
    RCLCPP_INFO(this->get_logger(), "Confidence Threshold: %f", confThreshold);
    RCLCPP_INFO(this->get_logger(), "Non-Maximum Suppression Threshold: %f", nmsThreshold);
    RCLCPP_INFO(this->get_logger(), "Input Width: %d", inpWidth);
    RCLCPP_INFO(this->get_logger(), "Input height: %d", inpHeight);
    RCLCPP_INFO(this->get_logger(), "Device: %s", device_.c_str());
}

void YoloObjectDetection::run_()
{   

    //Load the Darknet network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device_ == "cpu")
    {
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device_ == "gpu")
    {   
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
 
    try
    {
        //outputFile = "yolo_out_cpp.jpg";
        if (input_type_ == "image")
        {
            //Open image file
            ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_WARN(this->get_logger(),"No image found");
            }
            cap.open(input_path_);
            outputFile = output_file_;
            RCLCPP_INFO(this->get_logger(), "image opened successfully");
        }
        else if (input_type_ == "video")
        {
            //Open video file
            ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_WARN(this->get_logger(),"No video found");
            }
            cap.open(input_path_);
            outputFile = output_file_;
        }
    }
    catch(...)
    {
        RCLCPP_ERROR(this->get_logger(), "Could not open input file");
        rclcpp::shutdown();
    }
    
    // Get the video writer initialized to save the output video
    if (input_type_ == "video")
    {   
        double fps = cap.get(CAP_PROP_FPS);
        RCLCPP_INFO(this->get_logger(), "fps: %f", fps);
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), fps, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }

    //Create window
    static const string window_name = "YOLO Object Detection";
    namedWindow(window_name, WINDOW_NORMAL);

    //Process frames
    while (true)
    {
        // get frame from the video
        cap >> frame;
        //frame = imread(input_path_, IMREAD_COLOR);

        // Stop the program if reached end of video
        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "Done processing");
            if (write_output_)
            {
                RCLCPP_INFO(this->get_logger(), "Output writen to: %s", output_file_.c_str());
            }
            waitKey(3000);
            break;
        }

        // Create a 4D blob from a frames
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);
        
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Processing Time : %.2f ms", t);
        putText(frame, label, Point(0, frame.rows-15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        
        // Write the frame with the detection boxes
        if (write_output_)
        {
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);
            if (input_type_ == "image")
            {
                imwrite(outputFile, detectedFrame);
            }
            else
            {
                video.write(detectedFrame);
            }
        }

        //Display
        imshow(window_name, frame);
        char key = (char) waitKey(1);
        if (key == 27)
        {
            break;
            destroyAllWindows();
            rclcpp::shutdown();
        }
    }
    
    cap.release();
    if (input_type_ == "video") video.release();    
}

void YoloObjectDetection::postprocess(const Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int right = box.x + box.width;
        int bottom = box.y + box.height;

        drawPred(idx, classIds[idx], confidences[idx], left, top,
            right, bottom, frame);

        //drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 //box.x + box.width, box.y + box.height, frame);
    }
}

void YoloObjectDetection::drawPred(int idx, int classId, float conf, int left, int top, int right, int bottom, const Mat& frame)
{   
    //Load names of classes
    vector<string> classes;    
    //string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 65), 1); //Scalar(255, 178, 50)
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = "Tag:" + to_string(idx) + " " + classes[classId] + ":" + label; //
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1*labelSize.height)), Point(left + round(1*labelSize.width), top + baseLine), Scalar(0, 255, 65), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),1.5);
}

vector<String> YoloObjectDetection::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}