/*
 * DetectFace.h
 *
 *  Created on: 16 mei 2013
 *      Author: hansgaiser
 */

#ifndef DETECTFACE_H_
#define DETECTFACE_H_
 
#include <iomanip>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>

#include "MatlabIO.hpp"

#ifdef WITH_CUDA
#include <opencv2/gpu/gpu.hpp>
#endif

#include <tinyxml.h>

namespace eva
{

class DetectFace
{
public:
	DetectFace(const char * workingDir);

	struct Config
	{
		std::string faceCascadeName;
		std::string noseCascadeName;
		double scale;

		std::string imageTopic;
		bool showImages;
		bool cudaEnabled;
	};

	struct Model
	{
		std::string modelName;

		uint32_t width;
		uint32_t height;

		cv::Mat curr_points;

		cv::Mat size;

		cv::Mat shape_mean;
		cv::Mat shape_vectors;
		cv::Mat shape_transform;
		cv::Mat shape_mesh;

		cv::Mat app_mean;
		cv::Mat app_mean_mat;
		cv::Mat app_vectors;
		cv::Mat gradient;

		cv::Mat steepest_descent;
		cv::Mat H;
		cv::Mat invH;
		cv::Mat R;

		cv::Mat warp_map;
	};

	void spin();

private:
	std::string m_workingDir;
	std::string m_configPath;

	cv::CascadeClassifier m_faceCascade;
	cv::CascadeClassifier m_noseCascade;
#ifdef WITH_CUDA
	cv::gpu::CascadeClassifier_GPU m_faceCascadeGpu;
	cv::gpu::CascadeClassifier_GPU m_noseCascadeGpu;
#endif
	Config m_config;
	Model m_model;

	bool loadConfig();
	void processImage(cv::Mat image);

	void drawPoints(cv::Mat & image, cv::Mat points, cv::Scalar color);
	void drawMesh(cv::Mat & image, cv::Mat points, cv::Scalar color, cv::Mat mesh);
	void drawText(cv::Mat & image, std::string text, cv::Scalar fontColor);

	void matchModel(cv::Mat image);

	void loadModel();
};

} /* namespace eva */
#endif /* DETECTFACE_H_ */
