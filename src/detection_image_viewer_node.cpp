#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <perception_msgs/PointInImage.h>
#include <object_detection/Detection.h>
#include <object_detection/Detections.h>
#include <object_detection/Lidar_Point.h>
#include <object_detection/Lidar_Points.h>
#include <string>
#include <vector>
//#include <optional>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <iostream>
#include <ros/console.h>

using namespace std;

ros::Subscriber detection_sub;
image_transport::Publisher image_pub;
const int text_font = cv::FONT_HERSHEY_DUPLEX;
const double text_scale = 0.5;
const int text_thickness = 1;

//Config Parameters
string detection_topic, window_name;
bool flip_image;
int waitkey_delay, line_spacing;
bool draw_lidar;
double H_min, H_max, S, V, lidar_rad_min, lidar_rad_max, pose_rad;

//Running Variables
//cv_bridge::CvImageConstPtr cur_img_ptr;
std_msgs::Header target_header;
cv::Mat src_image, image;
object_detection::Detections detections;
bool stale_image, made_image, image_exists, image_found, in_sync, detections_found;
string cur_status;
cv::Size status_size;

//LIDAR
object_detection::Lidar_Points lidar_points;
bool lidar_found;

int write_line(cv::Mat& img, const string& str, cv::Point pos, cv::Scalar color){	
	int baseline = 0;
	cv::Size size = cv::getTextSize("0", text_font, text_scale, 1, &baseline);
	cv::putText(img, str, pos + cv::Point(0, size.height), text_font, text_scale, color);
	return size.height;
}

void render_image(){	
	image = src_image;

	if(lidar_found && draw_lidar){ //Disabled to diagnose a delay
		for(object_detection::Lidar_Point point : lidar_points.points){
			float rs = point.distance/lidar_points.max_distance;

			if(0 <= point.frame_x && point.frame_x <= 1 && 
				0 <= point.frame_y && point.frame_y <= 1) {

				cv::Point circle_center(point.frame_x*(image.cols - 1), point.frame_y*(image.rows - 1));
				int rad = round((lidar_rad_max - lidar_rad_min)*(1 - sqrt(rs)) + lidar_rad_min);
				
				int H = round(rs*(H_max - H_min) + H_min);
				
				double C = V*S;
				double X = C*(1 - abs(((H/60) % 2) - 1));
				double m = V - C;
				double r, g, b;
				if(0 <= H && H < 60){
					r = C;
					g = X;
					b = 0;
				}else if(60 <= H && H < 120){
					r = X;
					g = C;
					b = 0;
				}else if(120 <= H && H < 180){
					r = 0;
					g = C;
					b = X;
				}else if(180 <= H && H < 240){
					r = 0;
					g = X;
					b = C;
				}else if(240 <= H && H < 300){
					r = X;
					g = 0;
					b = C;
				}else if(300 <= H && H < 360){
					r = C;
					g = 0;
					b = X;
				}
				int R = round((r + m)*255);
				int G = round((g + m)*255);
				int B = round((b + m)*255);
				cv::Scalar point_color(B, G, R);
				
				cv::circle(image, circle_center, rad, point_color, cv::FILLED);
			}
		}
	}

	if(detections_found && lidar_found && (detections.num_detects > 0)){
		for(int i = 0; i < detections.num_detects; i++){
			int line_height = 0;
			object_detection::Detection det = detections.detections[i];
			cv::Scalar rect_color = (det.obj_class.compare("person") == 0) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 215, 255);
			if(det.stable){
				cv::Point c1 = cv::Point(det.box.xmin, det.box.ymin), c2 = cv::Point(det.box.xmax, det.box.ymax);
				cv::Point label_pos = cv::Point(det.box.xmin, det.box.ymin);

				cv::rectangle(image, c1, c2, rect_color);

				int oc_baseline = 0;
				cv::Size oc_size = cv::getTextSize(det.obj_class, text_font, text_scale, text_thickness, &oc_baseline);
				
				int id_baseline = 0;
				cv::Size id_size = cv::getTextSize(to_string(det.id), text_font, text_scale, text_thickness, &id_baseline);
				
				double max_height = max(oc_size.height, id_size.height);
				
				cv::rectangle(image, 
					cv::Point(det.box.xmin, det.box.ymin), 
					cv::Point(det.box.xmin + oc_size.width + id_size.width + 4*line_spacing, det.box.ymin - max_height - 2*line_spacing), 
					rect_color,
					cv::FILLED);
					
				cv::putText(image, 
					det.obj_class, 
					cv::Point(det.box.xmin + line_spacing, det.box.ymin - line_spacing - abs(max_height - oc_size.height)/2), 
					text_font, text_scale, cv::Scalar(0, 0, 0), text_thickness);
				
				cv::putText(image,
					to_string(det.id),
					cv::Point(det.box.xmin + 3*line_spacing + oc_size.width, det.box.ymin - line_spacing - abs(max_height - id_size.height)/2),
					text_font, text_scale, cv::Scalar(255, 255, 0), text_thickness);
			}
		}
	}

	if(flip_image){ cv::flip(image, image, 1); }

	write_line(image, to_string(detections.num_detects), cv::Point(0, status_size.height + 5), cv::Scalar(0, 255, 0));

	made_image = true;
	stale_image = false;
}

void send_image(){
	image_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
}

void detection_report_callback(const object_detection::Detections::ConstPtr& detections_msg){
	cv_bridge::CvImagePtr target_img_ptr = cv_bridge::toCvCopy(detections_msg->image, sensor_msgs::image_encodings::BGR8);
	target_header = detections_msg->image.header;
	src_image = target_img_ptr->image;
	lidar_points = detections_msg->lidar_points;
	detections = *detections_msg;
	image_found = true;
	stale_image = true;
	lidar_found = true;
	detections_found = true;
}

int main(int argc, char* argv[]) //Make this work with a constant rate loop, detections buffer, and header-matched images
{
	ros::init(argc, argv, "detection_image_viewer_node");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);

	int baseline = 0;
	cur_status = "";
	stale_image = false;
	made_image = false;
	image_found = false;
	lidar_found = false;
	in_sync = false;

	detection_topic = "/object_detector/detects";
	window_name = "ACTor targeting";
	flip_image = false;
	waitkey_delay = 10;
	line_spacing = 5;
	draw_lidar = false;
	
	H_min = 0;
	H_max = 240;
	S = 1;
	V = 1;
	lidar_rad_min = 1;
	lidar_rad_max = 2;

	nh.getParam("/detection_image_viewer_node/detection_topic", detection_topic);
	nh.getParam("/detection_image_viewer_node/window_name", window_name);
	nh.getParam("/detection_image_viewer_node/flip_image", flip_image);
	nh.getParam("/detection_image_viewer_node/waitkey_delay", waitkey_delay);
	nh.getParam("/detection_image_viewer_node/line_spacing", line_spacing);
	nh.getParam("/detection_image_viewer_node/draw_lidar", draw_lidar);

	detection_sub = nh.subscribe(detection_topic, 10, detection_report_callback);

	image_pub = it.advertise("/object_detector/targeting_image", 1);


	ros::Rate r(30);
	while(ros::ok()){
		//set_image();
		//set_detections();
		//set_lidar();
		if(stale_image && image_found){ render_image(); }
		if(made_image){ send_image(); }
		r.sleep();
		ros::spinOnce();
	}
}
