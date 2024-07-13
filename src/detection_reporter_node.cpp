#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <perception_msgs/LabeledPointInImage.h>
#include <perception_msgs/PointInImage.h>
#include <object_detection/Lidar_Point.h>
#include <object_detection/Lidar_Points.h>
#include <object_detection/Detection.h>
#include <object_detection/Detections.h>
#include <object_detection/BoundingBox.h>
#include <object_detection/BoundingBoxes.h>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <math.h>

#include <dynamic_reconfigure/server.h>
#include <object_detection/DetectionConfig.h>   

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;
const double PI = 3.14159265359;

ros::Publisher dist_pub, nolidar_pub;
ros::Subscriber box_sub, img_sub, lidar_sub;

//Config Parameters
string cam_topic, target_class, bounding_boxes_topic;
int buffer_size;
double smoothing_timeout, retarget_timeout;

//Running Variables
int cur_xres, cur_yres;
bool found_image;
cv_bridge::CvImageConstPtr cur_img_ptr;
std_msgs::Header cur_image_header;

// Bipartite Matching
double metric_threshold;
double persistence_timeout;
double speed_weight;
double speed_decay;
double position_decay;
double size_decay;
double pickup_time;
long current_id;
int current_closest_id;
std::vector<object_detection::Detection> current_detections;

typedef struct mapping_link{
	int from;
	int to;
	double metric;
} mapping_link;
bool link_compare(mapping_link l1, mapping_link l2){
	return (l1.metric < l2.metric);
}

// Lidar
vector<object_detection::Lidar_Points> lidar_buffer;
object_detection::Lidar_Points lidar_points;
bool lidar_found;
double center_yaw, center_pitch,
	hfov, vfov,
	min_radius, max_radius, 
	min_height, max_height, 
	camera_x, camera_y, camera_z;

struct lidar_point{
	double x, y, z;
	double pitch, yaw;
	double distance;
	double fx, fy;
};

object_detection::Lidar_Point compute_all(object_detection::Lidar_Point p0, double cx, double cy, double cz){
	object_detection::Lidar_Point tmp = p0;

	tmp.distance = sqrt(pow(p0.x - cx, 2) + pow(p0.y - cy, 2));
	tmp.pitch = atan((tmp.z - cz)/sqrt(pow(tmp.x - cx, 2) + pow(tmp.y - cy, 2)));
	tmp.yaw = atan((tmp.y - cy)/(tmp.x - cx));
	if(tmp.x <= cx){ tmp.yaw = tmp.yaw + PI; }
	else if(tmp.y <= cy){ tmp.yaw = tmp.yaw + 2*PI; }

	tmp.frame_x = 1 - fmod(tmp.yaw - center_yaw - hfov/2 + 2*PI, 2*PI)/hfov;
	tmp.frame_y = 1 - (tmp.pitch - center_pitch + vfov/2)/vfov;

	return tmp;
}

void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg){
	int rows = lidar_msg->height;
	int cols = lidar_msg->width;
	int x_offset = lidar_msg->fields[0].offset;
	int y_offset = lidar_msg->fields[1].offset;
	int z_offset = lidar_msg->fields[2].offset;
	int data_offset = lidar_msg->fields[3].offset;
	
	float xmax, xmin, ymax, ymin, zmax, zmin, rmax;
	
	vector<object_detection::Lidar_Point> points;
	for(int i = 0; i < rows*cols; i++){
		int ind = i*lidar_msg->point_step;
		float x, y, z, dist, angle, pitch;
		memcpy(&x, &lidar_msg->data[ind + x_offset], sizeof(float));
		memcpy(&y, &lidar_msg->data[ind + y_offset], sizeof(float));
		memcpy(&z, &lidar_msg->data[ind + z_offset], sizeof(float));
		if(i == 0){
			xmin = x;
			xmax = x;
			ymin = y;
			ymax = y;
			zmin = z;
			zmax = z;
			rmax = 0;
		}else{
			xmax = max(xmax, x);
			xmin = min(xmin, x);
			ymax = max(ymax, y);
			ymin = min(ymin, y);
			zmax = max(zmax, z);
			zmin = min(zmin, z);
		}

		object_detection::Lidar_Point tmp;
		tmp.x = x;
		tmp.y = y;
		tmp.z = z;
		object_detection::Lidar_Point computed = compute_all(tmp, camera_x, camera_y, camera_z);
		rmax = max(rmax, (float)abs(computed.distance));
		points.push_back(computed);
	}
	object_detection::Lidar_Points points_msg;
	points_msg.header = lidar_msg->header;
	points_msg.points = points;
	points_msg.max_distance = rmax;
	points_msg.xmin = xmin;
	points_msg.xmax = xmax;
	points_msg.ymin = ymin;
	points_msg.ymax = ymax;
	points_msg.zmin = zmin;
	points_msg.zmax = zmax;
	
	//Cludge
	points_msg.header.stamp = ros::Time::now();
	
	lidar_buffer.insert(lidar_buffer.begin(), points_msg);
	if(lidar_buffer.size() > buffer_size){ lidar_buffer.resize(buffer_size); }
}

void dynamic_reconfigure_callback(object_detection::DetectionConfig &config, uint32_t level){
	//ROS_INFO("Reconfigure Thing");
	speed_weight = config.speed_weight;
	speed_decay = config.speed_decay;
	position_decay = config.position_decay;
	size_decay = config.size_decay;
	metric_threshold = config.metric_threshold;
	
	// Lidar
	center_yaw = config.center_yaw;
	center_pitch = config.center_pitch;
	vfov = config.vfov;
	hfov = config.hfov;
	camera_x = config.camera_x;
	camera_y = config.camera_y;
	camera_z = config.camera_z;
	min_radius = config.min_radius;
	max_radius = config.max_radius;
	min_height = config.min_height;
	max_height = config.max_height;
}

bool pointsort(object_detection::Lidar_Point a, object_detection::Lidar_Point b){ return a.distance < b.distance; }

void set_lidar(ros::Time timestamp){
	long min_diff = -1;
	int min_ind = -1;
	for(int i = 0; i < lidar_buffer.size(); i++){
		long diff = abs((lidar_buffer[i].header.stamp - timestamp).toNSec());
		if(diff < min_diff || min_diff < 0){
			min_diff = diff;
			min_ind = i;
		}
	}
	if(min_ind >= 0){
		lidar_points = lidar_buffer[min_ind];
		lidar_found = true;
		ROS_INFO("Lidar Found!");
		//return;
	}else{
		lidar_found = false;
		ROS_INFO("Lidar Lost!");
	}
	//cout << "Diff: " << min_diff << endl;
	ROS_INFO("Diff: %d", min_diff);
}

double box_metric(object_detection::Detection base, object_detection::Detection test){
	double elapsed = (test.last_updated - base.last_updated).toSec();
	
	double x0 = base.x;
	double y0 = base.y;
	double dx0 = base.delta_x;
	double dy0 = base.delta_y;
	
	double px0 = x0 + elapsed*dx0;
	double py0 = y0 + elapsed*dy0;
	
	double x1 = test.x;
	double y1 = test.y;
	double dx1 = (x1 - x0)/elapsed;
	double dy1 = (y1 - y0)/elapsed;
	
	double xdiff = abs(dx0 - dx1);
	double ydiff = abs(dy0 - dy1);
	
	return sqrt(pow(px0 - x1, 2) + pow(py0 - y1, 2) + speed_weight*(pow(xdiff/(1 + xdiff), 2) + pow(ydiff/(1 + ydiff), 2)));
}

object_detection::Lidar_Point get_distance(double x0, double x1, double y0, double y1){
	vector<object_detection::Lidar_Point> points;
	for(object_detection::Lidar_Point point : lidar_points.points){
		double x = point.frame_x;
		double y = point.frame_y;
		if(x0 <= x && x <= x1 && y0 <= y && y <= y1){
			points.push_back(point);
		}
	}
	int size = points.size();
	if (size == 0){
		object_detection::Lidar_Point nah;
		nah.distance = -1;
		return nah; // Undefined
	}else{
		sort(points.begin(), points.end(), pointsort);
		object_detection::Lidar_Point avg;
		avg.x = 0;
		avg.y = 0;
		avg.z = 0;
		avg.distance = 0;
		avg.pitch = 0;
		avg.yaw = 0;
		avg.frame_x = 0;
		avg.frame_y = 0;
		for(int i = 0; i < min(10, size - 1); i++){
			avg.x += points[i].x/min(10, size - 1);
			avg.y += points[i].y/min(10, size - 1);
			avg.z += points[i].z/min(10, size - 1);
			avg.distance += points[i].distance/min(10, size - 1);
			avg.pitch += points[i].pitch/min(10, size - 1);
			avg.yaw += points[i].yaw/min(10, size - 1);
			avg.frame_x += points[i].frame_x/min(10, size - 1);
			avg.frame_y += points[i].frame_y/min(10, size - 1);
		}	
		return avg;
	}
}

object_detection::Detection make_detection(object_detection::BoundingBox box, ros::Time cur_time){
	double x0 = (double)box.xmin/cur_xres;
	double x1 = (double)box.xmax/cur_xres;
	double y0 = (double)box.ymin/cur_yres;
	double y1 = (double)box.ymax/cur_yres;	
	object_detection::Detection tmp_msg;

	tmp_msg.lidar_point = get_distance(x0, x1, y0, (y0 + y1)/2);

	tmp_msg.box = box;
	tmp_msg.width = (double)(box.xmax - box.xmin)/cur_xres;
	tmp_msg.height = (double)(box.ymax - box.ymin)/cur_yres;
	tmp_msg.y = (double)(box.ymin + box.ymax)/cur_yres - 1;
	tmp_msg.x = (double)(box.xmin + box.xmax)/cur_xres - 1;

	tmp_msg.first_seen = cur_time;
	tmp_msg.last_seen = cur_time;
	tmp_msg.last_updated = cur_time;
	tmp_msg.obj_class = box.label;
	return tmp_msg;
}

void publish_detections(){
	if(found_image){
		sensor_msgs::Image tmp_img_msg;
		cv_bridge::CvImage(std_msgs::Header(), "rgb8", cur_img_ptr->image).toImageMsg(tmp_img_msg);
		object_detection::Detections det_msg;
		
		det_msg.image_header = cur_image_header;
		det_msg.image = tmp_img_msg;
		det_msg.xres = cur_xres;
		det_msg.yres = cur_yres;
		
		//det_msg.lidar_points = lidar_points;
		
		det_msg.num_detects = current_detections.size();
		det_msg.detections = current_detections;

		det_msg.closest = det_msg.detections.size() <= 0 ? -1 : current_closest_id;

		nolidar_pub.publish(det_msg);
		
		object_detection::Detections det_with_lidar = det_msg;
		if(lidar_found){
			det_with_lidar.lidar_points = lidar_points;
		}
		dist_pub.publish(det_with_lidar);
		
	}
}

void box_callback(const object_detection::BoundingBoxes::ConstPtr& box_msg){
	ROS_INFO("Detection Callback: %d", (int)box_msg->bounding_boxes.size());
	//cout << "  - CALLBACK!" << endl;
	//set_lidar(box_msg->image_header.stamp);
	set_lidar(ros::Time::now());
	
	//ROS_INFO("Boxes: %f", box_msg->image_header.stamp.toSec());

	ros::Time cur_time = ros::Time::now();
	//set_lidar(cur_time);
	if(!lidar_found){ return; }
	int count = 0;

	// Process bounding boxes
	ROS_INFO("Process Bounding Boxes");
	std::vector<object_detection::Detection> tmp_detects;
	std::vector<mapping_link> mapping_links;
	for(object_detection::BoundingBox box : box_msg->bounding_boxes){
		//cout << "    - ISSABOX!" << endl;
		if(true || box.label.compare(target_class) == 0 || target_class.compare("") == 0){
			object_detection::Detection tmp_msg = make_detection(box, cur_time);
			for(int i = 0; i < current_detections.size(); i++){
				object_detection::Detection prev = current_detections[i];
				double pair_dist = box_metric(prev, tmp_msg);
				if((pair_dist <= metric_threshold) && (prev.obj_class.compare(tmp_msg.obj_class) == 0)){
					mapping_link tmp_link;
					tmp_link.from = i;
					tmp_link.to = count;
					tmp_link.metric = pair_dist;
					mapping_links.push_back(tmp_link);
				}else{
					//cout << "-- Disconnect: " << pair_dist << endl;
				}
			}
			
			tmp_detects.push_back(tmp_msg);
			count++;
			//cout << "      - I SEE!" << endl;
		}
	}
	
	//ROS_INFO("Update Matched Detections");
	// Update matched detections
	std::vector<bool> used_prev, used_next;
	for(int i = 0; i < current_detections.size(); i++){ used_prev.push_back(false); }
	for(int i = 0; i < tmp_detects.size(); i++){ used_next.push_back(false); }
	sort(mapping_links.begin(), mapping_links.end(), link_compare);
	for(int i = 0; i < mapping_links.size(); i++){
		mapping_link link = mapping_links[i];
		if(!(used_prev[link.from] || used_next[link.to])){
			object_detection::Detection prev = current_detections[link.from];
			object_detection::Detection next = tmp_detects[link.to];
			double elapsed = (next.last_updated - prev.last_updated).toSec();
			next.id = prev.id;
			next.first_seen = prev.first_seen;
			next.metric = link.metric;
			
			//double x0 = ((double)next.box.xmin + position_decay*prev.box.xmin)/(1 + position_decay);
			//double x1 = ((double)next.box.xmax + position_decay*prev.box.xmax)/(1 + position_decay);
			//double y0 = ((double)next.box.ymin + position_decay*prev.box.ymin)/(1 + position_decay);
			//double y1 = ((double)next.box.ymax + position_decay*prev.box.ymax)/(1 + position_decay);
			double x0 = ((double)next.box.xmin + position_decay*(prev.box.xmin + elapsed*prev.delta_x*cur_xres))/(1 + position_decay);
			double x1 = ((double)next.box.xmax + position_decay*(prev.box.xmax + elapsed*prev.delta_x*cur_xres))/(1 + position_decay);
			double y0 = ((double)next.box.ymin + position_decay*(prev.box.ymin + elapsed*prev.delta_y*cur_yres))/(1 + position_decay);
			double y1 = ((double)next.box.ymax + position_decay*(prev.box.ymax + elapsed*prev.delta_y*cur_yres))/(1 + position_decay);
			
			next.box.xmin = round(x0);
			next.box.xmax = round(x1);
			next.box.ymin = round(y0);
			next.box.ymax = round(y1);
			next.x = (x0 + x1)/cur_xres - 1;
			next.y = (y0 + y1)/cur_yres - 1;
			next.width = (x1 - x0)/cur_xres;
			next.height = (y1 - y0)/cur_yres;
			//next.x = (next.x + position_decay*(prev.x + elapsed*prev.delta_x))/(1 + position_decay);
			//next.y = (next.y + position_decay*(prev.y + elapsed*prev.delta_y))/(1 + position_decay);
			//next.width = (next.width + size_decay*prev.width)/(1 + size_decay);
			//next.height = (next.height + size_decay*prev.height)/(1 + size_decay);
			//next.box.xmin = cur_xres*round(next.x - next.width/2);
			//next.box.xmax = cur_xres*round(next.x + next.width/2);
			//next.box.ymin = cur_yres*round(next.y - next.height/2);
			//next.box.ymax = cur_yres*round(next.y + next.height/2);
			
			next.delta_x = ((next.x - prev.x)/elapsed + speed_decay*prev.delta_x)/(1 + speed_decay);
			next.delta_y = ((next.y - prev.y)/elapsed + speed_decay*prev.delta_y)/(1 + speed_decay);
			
			next.estimated = false;
			if((next.last_seen - next.first_seen) > ros::Duration(pickup_time)){
				next.stable = true;
			}
			
			current_detections[link.from] = next;
			used_prev[link.from] = true;
			used_next[link.to] = true;
		}
	}
	
	//ROS_INFO("Estimate Existing Detections");
	// Estimate existing detections that weren't matched
	for(int i = 0; i < used_prev.size(); i++){
		if(!used_prev[i]){
			object_detection::Detection det = current_detections[i];
			double elapsed = (cur_time - det.last_updated).toSec();
			det.x = det.x + elapsed*det.delta_x;
			det.y = det.y + elapsed*det.delta_y;
			det.box.xmin = det.box.xmin + elapsed*det.delta_x*cur_xres;
			det.box.xmax = det.box.xmax + elapsed*det.delta_x*cur_xres;
			det.box.ymin = det.box.ymin + elapsed*det.delta_y*cur_yres;
			det.box.ymax = det.box.ymax + elapsed*det.delta_y*cur_yres;
			
			det.last_updated = cur_time;
			det.estimated = true;
		}
	}
	
	//ROS_INFO("Add New Detections");
	// Add new detections that weren't matched
	for(int i = 0; i < used_next.size(); i++){
		if(!used_next[i]){
			object_detection::Detection det = tmp_detects[i];
			det.id = current_id;
			det.first_seen = cur_time;
			det.last_seen = cur_time;
			det.last_updated = cur_time;
			det.estimated = false;
			det.stable = false;
			current_detections.push_back(det);
			current_id = (current_id + 1) % 1024;
			//cout << "--  Add: " << det.id << " (" << count << ")" << endl;
		}
	}
	
	//ROS_INFO("Filter Old Detections");
	// Filter out old detections
	std::vector<object_detection::Detection> filtered_detections;
	std::copy_if(current_detections.begin(), current_detections.end(), std::back_inserter(filtered_detections), [=](object_detection::Detection det){
		//if((cur_time - det.last_seen) > ros::Duration(persistence_timeout)){ cout << "-- Drop: " << det.id << endl; }
		return (cur_time - det.last_seen) <= ros::Duration(persistence_timeout);
	});
	current_detections = filtered_detections;
	
	//ROS_INFO("Update State Variables");
	// Update state variables
	double tmp_closest_dist = -1;
	int tmp_closest_ind = -1;
	for(int i = 0; i < current_detections.size(); i++){
		object_detection::Detection det = current_detections[i];
		
		if(tmp_closest_ind == -1 || det.lidar_point.distance < tmp_closest_dist){
			tmp_closest_dist = det.lidar_point.distance;
			tmp_closest_ind = i;
		}
	}
	current_closest_id = tmp_closest_ind >= 0 ? current_detections[tmp_closest_ind].id : -1;
}

void img_callback(const sensor_msgs::Image::ConstPtr& img_msg){
	//ROS_INFO("Image: %f", img_msg->header.stamp.toSec());
	cur_img_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::RGB8);
	cur_xres = img_msg->width;
	cur_yres = img_msg->height;
	found_image = true;
}
/*
void sync_callback(const object_detection::BoundingBoxes::ConstPtr& box_msg, 
			const object_detection::Lidar_Points::ConstPtr& lidar_msg, 
			const sensor_msgs::Image::ConstPtr& img_msg){
	//ROS_INFO("Sync Callback Start");
	
	// Image Handling
	cur_img_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::RGB8);
	cur_xres = img_msg->width;
	cur_yres = img_msg->height;
	found_image = true;
	
	// Lidar Handling
	lidar_points = lidar_msg;
	lidar_found = true;
	
	// Detection Handling
	box_callback(box_msg);
	
	//ROS_INFO("Sync Callback End");
}*/

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "detection_reporter_node");
	ros::NodeHandle nh;

	cam_topic = "/object_detector/default_cam";
	target_class = "";
	cur_xres = -1;
	cur_yres = -1;
	found_image = false;
	lidar_found = false;
	buffer_size = 100;
	metric_threshold = 3;
	persistence_timeout = 0.5;
	pickup_time = 0.25;
	current_id = 0;
	current_closest_id = -1;
	speed_weight = 2;
	speed_decay = 0.5;
	position_decay = 0.1;
	size_decay = 0.0;
	
	// Lidar
	center_yaw = -PI/3;
	center_pitch = 0;
	hfov = PI/3;
	vfov = PI;
	camera_x = 0;
	camera_y = 0;
	camera_z = 0;
	min_radius = 0;
	max_radius = 0;
	min_height = 0;
	max_height = 0;
	
	nh.getParam("/object_detection_node/center_yaw", center_yaw);
	nh.getParam("/object_detection_node/center_pitch", center_pitch);
	nh.getParam("/object_detection_node/hfov", hfov);
	nh.getParam("/object_detection_node/vfov", vfov);
	nh.getParam("/object_detection_node/camera_x", camera_x);
	nh.getParam("/object_detection_node/camera_y", camera_y);
	nh.getParam("/object_detection_node/camera_z", camera_z);
	nh.getParam("/object_detection_node/min_radius", min_radius);
	nh.getParam("/object_detection_node/max_radius", max_radius);
	nh.getParam("/object_detection_node/min_height", min_height);
	nh.getParam("/object_detection_node/max_height", max_height);
	
	bounding_boxes_topic = "/object_detector/detects_raw";

	nh.getParam("/detection_reporter_node/cam_topic", cam_topic);
	nh.getParam("/detection_reporter_node/bounding_boxes_topic", bounding_boxes_topic);
	nh.getParam("/detection_reporter_node/target_class", target_class);
	nh.getParam("/detection_reporter_node/buffer_size", buffer_size);
	nh.getParam("/detection_reporter_node/metric_threshold", metric_threshold);
	nh.getParam("/detection_reporter_node/persistence_timeout", persistence_timeout);

	dist_pub = nh.advertise<object_detection::Detections>("/object_detector/detects", 10);
	nolidar_pub = nh.advertise<object_detection::Detections>("/object_detector/detects_nolidar", 10);
	box_sub = nh.subscribe(bounding_boxes_topic, 10, box_callback);
	img_sub = nh.subscribe(cam_topic, 10, img_callback);
	lidar_sub = nh.subscribe("/actor/pointcloud", 10, lidar_callback);
	
	// Synchronized Subscribers
	//message_filters::Subscriber<object_detection::BoundingBoxes> box_sub(nh, "/darknet_ros/bounding_boxes", 1);
	//message_filters::Subscriber<object_detection::Lidar_Points> lidar_sub(nh, "/object_detection/Lidar_points", 1);
	//message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, cam_topic, 1);
	
	//message_filters::TimeSynchronizer<object_detection::BoundingBoxes, object_detection::Lidar_Points, sensor_msgs::Image> sync(box_sub, lidar_sub, image_sub, 10);
	//typedef message_filters::sync_policies::ApproximateTime<object_detection::BoundingBoxes, object_detection::Lidar_Points, sensor_msgs::Image> DetectionSyncPolicy;
	//message_filters::Synchronizer<DetectionSyncPolicy> sync(DetectionSyncPolicy(10), box_sub, lidar_sub, image_sub);
	//sync.registerCallback(boost::bind(&sync_callback, _1, _2, _3));
	
	// Dynamic Reconfigure Stuff
	dynamic_reconfigure::Server<object_detection::DetectionConfig> server;
	dynamic_reconfigure::Server<object_detection::DetectionConfig>::CallbackType f;
	
	f = boost::bind(&dynamic_reconfigure_callback, _1, _2);
	server.setCallback(f);

	ros::Rate r(30);
	while(ros::ok()){
		publish_detections();
		r.sleep();
		ros::spinOnce();
	}
}
